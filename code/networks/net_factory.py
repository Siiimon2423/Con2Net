from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3, Con2Net_v1, Con2Net_v2, Con2Net_v3

def net_factory(net_type="unet", in_chns=3, class_num=5, feature_length=128):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == 'con2Net_v1':
        net = Con2Net_v1(in_chns=in_chns, class_num=class_num, feature_length=feature_length).cuda()
    elif net_type == 'con2Net_v2':
        net = Con2Net_v2(in_chns=in_chns, class_num=class_num, feature_length=feature_length).cuda()
    elif net_type == 'con2Net_v3':
        net = Con2Net_v3(in_chns=in_chns, class_num=class_num, feature_length=feature_length).cuda()
    return net
