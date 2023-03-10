from datasets.load_pre_made_dataset import PreMadeChangeDataset
from datasets.pointcloud import pointcloud, pointcloud_rgbimg
from datasets.vl_cmu_cd import vl_cmu_cd_eval, vl_cmu_cd_vpr
from datasets.pcd import gsv_eval, tsunami_eval,pcd_5fold, pcd_vpr
from datasets.changesim import changesim_eval
from datasets.cd2014 import cd2014
from datasets.kitti_moseg_extended import kitti_moseg_extended
from datasets.desk import desk_demo
import os
import torch
from torch.utils.data import DataLoader

def prepare_trainval(args,
                     source_img_transforms,target_img_transforms,
                     flow_transform,co_transform,change_transform):
    if args.test_only: return None, None

    train_datasets, val_datasets = {},{}

    train_synthetic_dataset, val_synthetic_dataset = PreMadeChangeDataset(root=os.path.join(args.training_data_dir,'synthetic_dataset'),
                                                                    source_image_transform=source_img_transforms,
                                                                    target_image_transform=target_img_transforms,
                                                                    flow_transform=flow_transform,
                                                                    co_transform=None,
                                                                    change_transform=change_transform,
                                                                    split=args.split_ratio,
                                                                    split2=args.split2_ratio,
                                                                    multi_class=args.multi_class)  # train:val = 95:5

    if 'synthetic' in args.trainset_list:
        train_datasets['synthetic'] = train_synthetic_dataset
    if 'synthetic' in args.valset_list:
        val_datasets['synthetic'] = val_synthetic_dataset


    if 'vl_cmu_cd' in args.trainset_list:
        train_datasets['vl_cmu_cd'] = vl_cmu_cd_eval(root=os.path.join(args.evaluation_data_dir, 'VL-CMU-CD'),
                                                     source_image_transform=source_img_transforms,
                                                     target_image_transform=target_img_transforms,
                                                     change_transform=change_transform,
                                                     split='train',
                                                     img_size=args.train_img_size
                                                     )
    if 'pcd' in args.trainset_list:
        train_datasets['pcd'] =pcd_5fold(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
                                      source_image_transform=source_img_transforms,
                                      target_image_transform=target_img_transforms,
                                      change_transform=change_transform,
                                      split= 'train',
                                      img_size = args.train_img_size
                                      )
    if 'cd2014' in args.trainset_list:
        train_datasets['cd2014'] =cd2014(root=os.path.join(args.evaluation_data_dir,'CD2014'),
                                      source_image_transform=source_img_transforms,
                                      target_image_transform=target_img_transforms,
                                      change_transform=change_transform,
                                      split= 'train',
                                      img_size = args.train_img_size
                                      )
        
    if 'pointcloud' in args.trainset_list:
        droot = '/media/rit/SSD1TB/Scenario1-3D-Change-Detection/Dataset/Localization/'
        train_datasets['pointcloud'] = pointcloud(root=droot,
                                      source_image_transform=source_img_transforms,
                                      target_image_transform=target_img_transforms,
                                      change_transform=change_transform,
                                      multi_class=args.multi_class,
                                      mapname='*ouse',
                                      seqname='Seq_[0-1]',
                                      img_size= args.train_img_size
                                      )
        
    if 'pointcloud_rgbimg' in args.trainset_list:
        droot = '/media/rit/SSD1TB/Scenario1-3D-Change-Detection/Dataset/Localization/'
        train_datasets['pointcloud_rgbimg'] = pointcloud_rgbimg(root=droot,
                                      source_image_transform=source_img_transforms,
                                      target_image_transform=target_img_transforms,
                                      change_transform=change_transform,
                                      multi_class=args.multi_class,
                                      mapname='*ouse',
                                      seqname='Seq_[0-1]',
                                      img_size= args.train_img_size
                                      )

    if 'changesim_normal' in args.trainset_list:
        train_datasets['changesim_normal'] = changesim_eval(root=os.path.join(args.training_data_dir,'Query_Seq_Train'),
                                      source_image_transform=source_img_transforms,
                                      target_image_transform=target_img_transforms,
                                      change_transform=change_transform,
                                      multi_class=args.multi_class,
                                      mapname='*',
                                      seqname='Seq_[0-1]',
                                      img_size= args.train_img_size,
                                                            vpr_candidates=args.vpr_candidates,
                                                            vpr_patchnetvlad=args.vpr_patchnetvlad
                                                            )
    if 'changesim_dust' in args.trainset_list:
        train_datasets['changesim_dust'] = changesim_eval(root=os.path.join(args.training_data_dir,'Query_Seq_Train'),
                                      source_image_transform=source_img_transforms,
                                      target_image_transform=target_img_transforms,
                                      change_transform=change_transform,
                                      multi_class=args.multi_class,
                                      mapname='*',
                                      seqname='Seq_[0-1]_dust',
                                      img_size= args.train_img_size,
                                      vpr_candidates=args.vpr_candidates,
                                      vpr_patchnetvlad=args.vpr_patchnetvlad
                                      )
    if 'changesim_dark' in args.trainset_list:
        train_datasets['changesim_dark'] = changesim_eval(root=os.path.join(args.training_data_dir,'Query_Seq_Train'),
                                      source_image_transform=source_img_transforms,
                                      target_image_transform=target_img_transforms,
                                      change_transform=change_transform,
                                      multi_class=args.multi_class,
                                      mapname='*',
                                      seqname='Seq_[0-1]_dark',
                                      img_size= args.train_img_size,
                                                          vpr_candidates=args.vpr_candidates,
                                                          vpr_patchnetvlad=args.vpr_patchnetvlad
                                                          )
    if 'tsunami' in args.trainset_list:
        train_datasets['tsunami'] = tsunami_eval(root=os.path.join(args.evaluation_data_dir, 'TSUNAMI'),
                                                      source_image_transform=source_img_transforms,
                                                      target_image_transform=target_img_transforms,
                                                      change_transform=change_transform,
                                                split='train',
                                        img_size = args.train_img_size

                                        )
    if 'gsv' in args.trainset_list:
        train_datasets['gsv'] = gsv_eval(root=os.path.join(args.evaluation_data_dir, 'GSV'),
                                        source_image_transform=source_img_transforms,
                                        target_image_transform=target_img_transforms,
                                        change_transform=change_transform,
                                            split='train',
                                         img_size=args.train_img_size

                                         )
    if 'kitti' in args.trainset_list:
        train_datasets['kitti'] = kitti_moseg_extended(root=os.path.join(args.evaluation_data_dir, 'KITTI_MOD_extended'),
                                        source_image_transform=source_img_transforms,
                                        target_image_transform=target_img_transforms,
                                        change_transform=change_transform,
                                        split='train',
                                         img_size=args.train_img_size
                                         )
    print('-------------------------------------------------------------')

    for k, d in train_datasets.items():
        print('- LOADING train split of {} ({} pairs)'.format(k,len(d)))

    train_dataset = torch.utils.data.ConcatDataset([ d for k,d in train_datasets.items()])
    print('# of training samples in total: ({} pairs)'.format(len(train_dataset)))

    for k, d in val_datasets.items():
        print('- LOADING val split of {} ({} pairs)'.format(k,len(d)))

    val_dataset = torch.utils.data.ConcatDataset([ d for k,d in val_datasets.items()])
    print('# of validataion samples in total: ({} pairs)'.format(len(val_dataset)))
    print('-------------------------------------------------------------')

    # Dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_threads,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.n_threads)

    return train_dataloader, val_dataloader

def prepare_test(args,source_img_transforms,target_img_transforms,flow_transform,co_transform,change_transform):

    test_datasets = {}

    for testset in args.testset_list:
        if testset == 'vl_cmu_cd':
            test_datasets['vl_cmu_cd'] = vl_cmu_cd_eval(root=os.path.join(args.evaluation_data_dir,'VL-CMU-CD'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=source_img_transforms,
                                          change_transform=change_transform,
                                          split='test',
                                          img_size=args.test_img_size,
                                          vpr_candidates = args.vpr_candidates,
                                          vpr_patchnetvlad = args.vpr_patchnetvlad,
                                          vpr_netvlad = args.vpr_netvlad
                                                        )
        elif testset == 'pcd':
            test_datasets['pcd'] =pcd_5fold(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          split= 'test',
                                          tsunami_or_gsv='both',
                                            img_size=args.test_img_size
                                            )
        elif testset == 'tsunami_vpr':
            test_datasets['tsunami_vpr'] =pcd_vpr(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          split= 'test',
                                          tsunami_or_gsv='tsunami',
                                            img_size=args.test_img_size
                                            )
        elif testset == 'gsv_vpr':
            test_datasets['gsv_vpr'] =pcd_vpr(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          split= 'test',
                                          tsunami_or_gsv='gsv',
                                            img_size=args.test_img_size
                                            )
        elif testset == 'tsunami':
            test_datasets['tsunami'] =pcd_5fold(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          split= 'test',
                                            tsunami_or_gsv='tsunami',

                                                img_size=args.test_img_size
                                            )
        elif testset == 'gsv':
            test_datasets['gsv'] =pcd_5fold(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          split= 'test',
                                            tsunami_or_gsv='gsv',
                                            img_size=args.test_img_size
                                            )
        elif testset == 'cd2014':
            test_datasets['cd2014'] = cd2014(root=os.path.join(args.evaluation_data_dir, 'CD2014'),
                                              source_image_transform=source_img_transforms,
                                              target_image_transform=target_img_transforms,
                                              change_transform=change_transform,
                                              split='test',
                                              img_size=args.train_img_size
                                              )
        elif 'pointcloud' in args.trainset_list:
            droot = '/media/rit/SSD1TB/Scenario1-3D-Change-Detection/Dataset/Localization/'
            test_datasets['pointcloud'] = pointcloud(root=droot,
                                        source_image_transform=source_img_transforms,
                                        target_image_transform=target_img_transforms,
                                        change_transform=change_transform,
                                        multi_class=args.multi_class,
                                        mapname='Storage',
                                        seqname='Seq_[0-1]',
                                        img_size= args.test_img_size
                                        )
        
        elif 'pointcloud_rgbimg' in args.trainset_list:
            droot = '/media/rit/SSD1TB/Scenario1-3D-Change-Detection/Dataset/Localization/'
            test_datasets['pointcloud_rgbimg'] = pointcloud_rgbimg(root=droot,
                                        source_image_transform=source_img_transforms,
                                        target_image_transform=target_img_transforms,
                                        change_transform=change_transform,
                                        multi_class=args.multi_class,
                                        mapname='Storage',
                                        seqname='Seq_[0-1]',
                                        img_size= args.test_img_size
                                        )
        elif testset == 'changesim_normal':
            test_datasets['changesim_normal'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'Query_Seq_Test'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          multi_class=args.multi_class,
                                          mapname='*',
                                          # seqname='Seq_[0-1]',
                                                               seqname = 'Seq_0',
                                                               img_size=args.test_img_size,
                                          vpr_candidates = args.vpr_candidates,
                                          vpr_patchnetvlad= args.vpr_patchnetvlad,
                                          vpr_netvlad=args.vpr_netvlad
                                                               )
        elif testset == 'changesim_dust':
            test_datasets['changesim_dust'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'Query_Seq_Test'),
                                              source_image_transform=source_img_transforms,
                                              target_image_transform=target_img_transforms,
                                              change_transform=change_transform,
                                              multi_class=args.multi_class,
                                              mapname='*',
                                              # seqname='Seq_[0-1]_dust',
                                                             seqname='Seq_0_dust',

                                                             img_size=args.test_img_size,
                                                             vpr_candidates=args.vpr_candidates,
                                                             vpr_patchnetvlad=args.vpr_patchnetvlad,
                                          vpr_netvlad=args.vpr_netvlad

                                                             )
        elif testset == 'changesim_dark':
            test_datasets['changesim_dark'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'Query_Seq_Test'),
                                              source_image_transform=source_img_transforms,
                                              target_image_transform=target_img_transforms,
                                              change_transform=change_transform,
                                              multi_class=args.multi_class,
                                              mapname='*',
                                              # seqname='Seq_[0-1]_dark',
                                                             seqname='Seq_0_dark',

                                                             img_size=args.test_img_size,
                                                             vpr_candidates=args.vpr_candidates,
                                                             vpr_patchnetvlad=args.vpr_patchnetvlad,
                                          vpr_netvlad=args.vpr_netvlad

                                                             )
        elif testset == 'tunnel_normal':
            test_datasets['tunnel_normal'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'Tunnel'),
                                              source_image_transform=source_img_transforms,
                                              target_image_transform=target_img_transforms,
                                              change_transform=change_transform,
                                              multi_class=args.multi_class,
                                              mapname='*',
                                              seqname='Seq_0',
                                                            img_size=args.test_img_size

                                                            )
        elif testset == 'tunnel_dust':
            test_datasets['tunnel_dust'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'Tunnel'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          multi_class=args.multi_class,
                                          mapname='*',
                                          seqname='Seq_0_dust',
                                                          img_size=args.test_img_size

                                                          )
        elif testset == 'tunnel_dark':
            test_datasets['tunnel_dark'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'Tunnel'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          multi_class=args.multi_class,
                                          mapname='*',
                                          seqname='Seq_0_dark',
                                          img_size = args.test_img_size
                                          )
        elif testset == 'tunnel_fire':
            test_datasets['tunnel_fire'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'Tunnel'),
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          change_transform=change_transform,
                                          multi_class=args.multi_class,
                                          mapname='*',
                                          seqname='Seq_0_fire',
                                          img_size = args.test_img_size
                                          )
        elif testset == 'kitti':
            test_datasets['kitti'] = kitti_moseg_extended(
                root=os.path.join(args.evaluation_data_dir, 'KITTI_MOD_extended'),
                source_image_transform=source_img_transforms,
                target_image_transform=target_img_transforms,
                change_transform=change_transform,
                split='test',
                img_size=args.test_img_size
                )
        elif testset == 'tunnel_real':
            test_datasets['tunnel_real'] = desk_demo(
                root=os.path.join(args.evaluation_data_dir, 'RealTunnel_jh'),
                source_image_transform=source_img_transforms,
                target_image_transform=target_img_transforms,
                img_size=args.test_img_size
                )
    total_len = 0
    for k, d in test_datasets.items():
        print('- LOADING test split of {} ({} pairs)'.format(k,len(d)))
        total_len+=len(d)
    print('# of test samples in total: ({} pairs)'.format(total_len))
    print('-------------------------------------------------------------')

    test_dataloaders = {k:DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=False,num_workers=args.n_threads)
                        for k, test_dataset in test_datasets.items()}

    return test_dataloaders