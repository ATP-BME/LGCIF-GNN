import sklearn
from sklearn.metrics import roc_auc_score,roc_curve,auc,accuracy_score,precision_recall_fscore_support,confusion_matrix
import matplotlib.pyplot as plt
from torch import nn, softmax
from model.model_local_global import LGMF_GNN
from dataloader_local_global import dataloader_lg,prepare_local_dataloader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from opt import *
from utils.metrics import accuracy, auc, prf
from utils.loss import FocalLoss
from scipy.special import softmax
from data.dataprocess import * 
import random
import yaml
from torch.optim import lr_scheduler

with open(opt.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)
print(config)

if __name__ == '__main__':
    opt = OptInit().initialize()
    OptInit().print_args()
    print('  Loading dataset ...')
    local_dataloader_pre,sub_IDs = prepare_local_dataloader([config['data']['time_seires'][0]],config['data']['t1_root'])
    # local_dataloader_post,sub_IDs = prepare_local_dataloader([config['data']['time_seires'][1]],config['data']['t1_root'])


    if 'treatment' in config['data']['dataset']: 
        balanced_names = np.genfromtxt("H:/treatment/rest_final/subIDs_all.txt", dtype=str)



    num_before = len(sub_IDs)
    sub_IDs = [name  for index,name in enumerate(sub_IDs) if name in balanced_names]
    assert len(sub_IDs) == num_before
    sub_IDs = np.array(sub_IDs)
    print('case num:',len(sub_IDs))
    
    dl = dataloader_lg()
    sub_IDs,y,nonimg,site_num,y_site = dl.load_data(subject_IDs=sub_IDs)
    n_sub = len(sub_IDs)
    np.save('sub_IDs.npy',sub_IDs)
    
    unique_IDs = np.unique(sub_IDs,return_index=False) # S+R+A=2406

    # 10 fold cv
    n_folds = 5
    train_inds,val_inds = dl.data_split(n_folds,train_val_num=len(unique_IDs))

    # loso cv
    # train_inds,val_inds = dl.data_split_loso()
    # n_folds = len(train_inds)

    # out test + loso cv
    # train_inds,val_inds = dl.data_split_site()
    # n_folds = len(train_inds)

    # print(cv_splits)
    global mean_tpr
    global mean_fpr
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    global cnt
    cnt = 0
    corrects_val = np.zeros(n_folds, dtype=np.int32)
    accs_val = np.zeros(n_folds, dtype=np.float32)
    aucs_val = np.zeros(n_folds, dtype=np.float32)
    prfs_val = np.zeros([n_folds, 3], dtype=np.float32)

    corrects_test = np.zeros(n_folds, dtype=np.int32)
    accs_test = np.zeros(n_folds, dtype=np.float32)
    aucs_test = np.zeros(n_folds, dtype=np.float32)
    prfs_test = np.zeros([n_folds, 3], dtype=np.float32)

    best_accs_test = np.zeros(n_folds, dtype=np.float32)
    best_prfs_test = np.zeros([n_folds, 3], dtype=np.float32)

    opt.ckpt_path = os.path.join(opt.ckpt_path,opt.mode)


    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))

        ### fold cv
        # train_name = unique_IDs[train_inds[fold]]
        # val_name = unique_IDs[val_inds[fold]]
        # test_name = val_name
        # train_ind = [index  for index,name in enumerate(sub_IDs) if name in train_name]
        # val_ind = [index  for index,name in enumerate(sub_IDs) if name in val_name]
        # test_ind = val_ind

        # loso cv
        train_ind = train_inds[fold]
        val_ind = val_inds[fold]
        test_ind = val_ind

        ### cv val
        # train_val_name = unique_IDs[train_inds[fold]]
        # test_name = unique_IDs[val_inds[fold]]


        wrong = [ind for ind in test_ind if ind in train_ind]
        print(wrong)


        # print('val_ind',val_ind)
        print('train HC:MDD =',sum(y[train_ind]==0),':',sum(y[train_ind]==1))
        print('val HC:MDD =',sum(y[val_ind]==0),':',sum(y[val_ind]==1))
        print('test HC:MDD =',sum(y[test_ind]==0),':',sum(y[test_ind]==1))
        train_HC_ind = np.array([ind for ind in np.argwhere(y==1) if ind in train_ind])
        train_MDD_ind = np.array([ind for ind in np.argwhere(y==0) if ind in train_ind])


        if torch.cuda.is_available():
            torch.cuda.manual_seed(n_folds)

        np.random.seed(n_folds)  # Numpy module.
        random.seed(n_folds)

        config['train']["seq_len"] = config['data']['window_width']

        print('using pre only')
        model = LGMF_GNN(nonimg,
                         site_num = site_num,
                         roi_num = config['train']['node_size'],
                         local_fea_dim = config['train']['node_size'],
                         global_fea_dim = config['train']['node_size']*config['model']['embedding_size'],
                         timeseries_len = config['train']["seq_len"],
                         local_dataloader = local_dataloader_pre,
                         train_HC_ind=train_HC_ind,
                         train_MDD_ind=train_MDD_ind,
                         test_ind=test_ind)
        model = model.to(opt.device)

        # build loss, optimizer, metric
        if not opt.focal_loss:
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        if opt.optimizer=='Adam':
            print('optimize together')
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
            
        
        if opt.optimizer=='RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(),lr=opt.lr,weight_decay=opt.wd)
        
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

        labels = torch.as_tensor(y, dtype=torch.long).to(opt.device)
        
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)
        writer = SummaryWriter(log_dir='./tb_logs/fold_{}'.format(fold))


        def plot_embedding(data, label, title):
            plt.figure()
            x_min, x_max = np.min(data, 0), np.max(data, 0)
            data = (data - x_min) / (x_max - x_min)
            p = [[0] for _ in range(10)]
            p2 = [[0] for _ in range(10)]
            for i in range(len(label)):
                if label[i] == 0:
                    p = plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#FFD700')#, alpha=0.8
                elif label[i] == 1:
                    p2 = plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#800080')
            plt.legend((p, p2), ('HC', 'MDD'))
            plt.savefig('./draw_figure/MDD/Result{:d}.png'.format(fold), dpi=600)

        def train(writer):
            print("  Number of training samples %d" % len(train_ind))
            print("  Start training...\r\n")
            acc = 0
            auc_best = 0
            best_k = 0
            for epoch in range(opt.num_iter):
                model.train()
                # optimaize together
                optimizer.zero_grad()
                # # cycle optimize
                # optimizer_G.zero_grad()
                # optimizer_L.zero_grad()

                scheduler.step()
                with torch.set_grad_enabled(True):
                    node_logits,shift_loss, att, emb1, com1, com2, emb2,k_num,local_loss,local_site_loss,local_acc,local_site_acc,_ = model.forward(dl,train_ind)
                    
                    loss_class = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss_dep = (loss_dependence(emb1, com1, n_sub)
                                + loss_dependence(emb2, com2, n_sub)) / 2

                   
                    loss_com = common_loss2(com1, com2)

                    
                    
                    loss_class = opt.g_closs_w*loss_class
                    loss_dep = opt.dep_loss_w * loss_dep
                    loss_com = opt.com_loss_w * loss_com
                    local_loss = opt.l_closs_w * local_loss
                    shift_loss = opt.shift_loss_weight * shift_loss
                    local_site_loss = opt.l_sloss_w * local_site_loss

                    loss = loss_class  + loss_dep + loss_com  
                    if opt.use_site_loss:
                        loss = loss + local_site_loss
                    if opt.shift_robust:
                        loss = loss + shift_loss 
                    if opt.use_local_loss:
                        loss = loss + local_loss 
                   
                    loss.backward()
                    optimizer.step()

                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                model.eval()
                with torch.set_grad_enabled(False):

                    node_logits,shift_loss, att,emb1, com1, com2, emb2,k_num,_,_,_,_,_ = model.forward(dl)

                logits_test = node_logits[val_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[val_ind])

                auc_test = auc(logits_test, y[val_ind])
                prf_test = prf(logits_test, y[val_ind])

                writer.add_scalar('AUC/test',auc_test,epoch)
                writer.add_scalar('G_Acc/train',acc_train,epoch)
                writer.add_scalar('G_Acc/test',acc_test,epoch)
                writer.add_scalar('L_Acc/class',local_acc,epoch)
                writer.add_scalar('L_Acc/site',local_site_acc,epoch)
                writer.add_scalar('Loss/l_closs',local_loss.item(),epoch)
                writer.add_scalar('Loss/l_sloss',local_site_loss.item(),epoch)
                writer.add_scalar('Loss/shift_loss',shift_loss.item(),epoch)
                writer.add_scalar('Loss/g_closs',loss_class.item(),epoch)
                writer.add_scalar('Loss/dep_loss',loss_dep.item(),epoch)
                writer.add_scalar('Loss/com_loss',loss_com.item(),epoch)

                if epoch % 20 == 0:
                    pos_probs = softmax(logits_test, axis=1)[:, 1]
                    
                    fpr,tpr,thresholds =roc_curve(y[val_ind],pos_probs)
                    auc_plot =roc_auc_score( y[val_ind],pos_probs)
                    plt.plot(fpr,tpr)
                    plt.title("auc=%.4f"%(auc_plot))
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.fill_between(fpr,tpr,where=(tpr>0),color='green',alpha=0.5)
                    # plt.show()
                    plt.draw()
                    if not os.path.exists('./fig/fold_%d'%(fold)):
                        os.makedirs('./fig/fold_%d'%(fold))
                    if epoch % 100 == 0:
                        plt.savefig('./fig/fold_%d/epoch_%d.png'%(fold,epoch))
                    plt.close()

                    for param_group in optimizer.param_groups:
                        current_lr=param_group['lr']
                    print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f},\tval_acc:{:.5f},\tlr:{:.5f},\tlocal_acc:{:.5f},\tlocal_site_acc:{:.5f},\tlocal_cls_loss:{:.5f},\tlocal_site_loss:{:.5f},\tshift_loss:{:.5f},\tloss_class: {:.5f},\t\tloss_dep: {:.5f}, \tloss_com: {:.5f}".format(epoch, loss.item(), acc_train.item(),acc_test,current_lr,local_acc,local_site_acc,local_loss.item(),local_site_loss.item(),shift_loss.item(),loss_class.item(),loss_dep.item(),loss_com.item()))
                   
                if auc_test >= auc_best and epoch > 9:
                    acc = acc_test
                    auc_best = auc_test
                    correct = correct_test
                    aucs_val[fold] = auc_test
                    prfs_val[fold] = prf_test
                    best_k = k_num
                    if opt.ckpt_path != '':
                        if not os.path.exists(opt.ckpt_path):
                            # print("Checkpoint Directory does not exist! Making directory {}".format(opt.ckpt_path))
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs_val[fold] = acc
            corrects_val[fold] = correct
            print("\r\n => Fold {} val accuacry {:.5f},best knn num:{}".format(fold, acc,best_k))



        def evaluate(infer_best_n=None):
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            global cnt
            global mean_tpr
            model.load_state_dict(torch.load(fold_model_path))
            print('model loaded:',fold_model_path)
            
            model.eval()
            
            if infer_best_n is not None:
                node_logits,shift_loss, att,emb1, com1, com2, emb2,_,_,_,_,_,_ = model.forward(dl,n=infer_best_n[fold])
            else:
                node_logits,shift_loss, att,emb1, com1, com2, emb2,_,_,_,_,_,_ = model.forward(dl)

            logits_test = node_logits[test_ind].detach().cpu().numpy()
            # save result
            test_result = {'logits':logits_test,'labels':y[test_ind]}

            preds_test = node_logits[test_ind].max(1)[1].detach().cpu().numpy()
            cm = confusion_matrix(y[test_ind].reshape(-1,1),preds_test.reshape(-1,1))
            corrects_test[fold], accs_test[fold] = accuracy(logits_test, y[test_ind])
            aucs_test[fold] = auc(logits_test, y[test_ind])
            prfs_test[fold] = prf(logits_test, y[test_ind])
            cnt += 1
            pos_probs = softmax(logits_test, axis=1)[:, 1]
            fpr, tpr, thresholds = roc_curve(y[test_ind], pos_probs)
            # best threshold
            maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
            best_threshold = thresholds[maxindex]
            result = np.zeros(pos_probs.shape)
            result[pos_probs>=best_threshold] = 1
            result[pos_probs<best_threshold] = 0
            best_acc = accuracy_score(y[test_ind],result)
            best_cm = confusion_matrix(y[test_ind].reshape(-1,1),result.reshape(-1,1))
            p,r,f,s = precision_recall_fscore_support(y[test_ind].reshape(-1,1),result.reshape(-1,1),average='binary')
            best_prfs_test[fold] = [p,r,f]
            best_accs_test[fold] = best_acc
            
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            lw = 2
            plt.plot(fpr, tpr, lw=lw, label='ROC fold {0:d} curve (area= {1:.2f})'.format(cnt, roc_auc))
            plt.title("auc=%.4f"%(roc_auc))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.fill_between(fpr,tpr,where=(tpr>0),color='green',alpha=0.5)
            plt.draw()
            if not os.path.exists('./fig/fold_%d'%(fold)):
                os.makedirs('./fig/fold_%d'%(fold))
            plt.savefig('./fig/fold_%d/test.png'%(fold))
            plt.close()

            print(cm)
            print("  Fold {} test accuracy {:.5f},best test accuracy {:.5f}, AUC {:.5f}, best threshold {:.5f}".format(fold, accs_test[fold],best_acc, aucs_test[fold],best_threshold))
            print(best_cm)
            return test_result




        if opt.train == 1:
            train(writer)
            evaluate()
        elif opt.train == 0:
            infer_best_n = [10,10,10,10,10]
            evaluate(infer_best_n)

    if opt.train == 1:
        print("\r\n========================== Finish train ==========================")
        print(accs_val)
        print(aucs_val)
        print(prfs_val)
        n_samples = (len(val_ind)*n_folds)
        acc_nfold = np.sum(corrects_val) / (len(val_ind)*n_folds)
        print("=> Average val accuracy in {}-fold CV: {:.5f}({:.5f})".format(n_folds, acc_nfold,np.std(accs_val)))
        print("=> Average val accuracy in {}-fold CV: {:.5f}({:.5f})".format(n_folds, np.mean(accs_val),np.std(accs_val)))
        print("=> Average val AUC in {}-fold CV: {:.5f}({:.5f})".format(n_folds, np.mean(aucs_val),np.std(aucs_val)))
        se, sp, f1 = np.mean(prfs_val, axis=0)
        se_, sp_, f1_ = np.std(prfs_val, axis=0)
        print("=> Average val precision {:.5f}({:.5f}), recall {:.5f}({:.5f}), F1-score {:.5f}({:.5f})".format(se,se_, sp,sp_, f1,f1_))

    print("\r\n========================== Finish test ==========================")
    print(accs_test)
    print(aucs_test)
    print(prfs_test)
    n_samples = (len(test_ind)*n_folds)
    acc_nfold = np.sum(corrects_test) / (len(test_ind)*n_folds)
    best_acc_nfold = np.mean(best_accs_test)
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.5f}); Best test accuracy {:.5f}({:.5f})".format(n_folds, acc_nfold,np.std(accs_test),best_acc_nfold,np.std(best_accs_test)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.5f})".format(n_folds, np.mean(aucs_test),np.std(aucs_test)))
    se, sp, f1 = np.mean(prfs_test, axis=0)
    se_, sp_, f1_ = np.std(prfs_test, axis=0)
    print("=> Average test precision {:.5f}({:.5f}), recall {:.5f}({:.5f}), F1-score {:.5f}({:.5f})".format(se,se_, sp,sp_, f1,f1_))
    best_se, best_sp, best_f1 = np.mean(best_prfs_test, axis=0)
    best_se_, best_sp_, best_f1_ = np.std(best_prfs_test, axis=0)
    print("=> Best average test precision {:.5f}({:.5f}), recall {:.5f}({:.5f}), F1-score {:.5f}({:.5f})".format(best_se,best_se_, best_sp,best_sp_, best_f1,best_f1_))

