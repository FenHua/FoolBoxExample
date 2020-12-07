import torch
import torchvision.models as models
from foolbox.attacks import LinfPGD, L2PGD
from foolbox import PyTorchModel, accuracy, samples


def main() -> None:
    # 初始化一个分类模型
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)  # bounds用于将输入归一化到指定范围内，preprocessing 对输入进行归一化处理
    images, labels = samples(fmodel, dataset="imagenet", batchsize=16)        # 获取数据
    labels = torch.tensor(labels, dtype=torch.long)    # 数据类型的转换
    images = images.cuda()                             # GPU化
    labels = labels.cuda()                             # GPU化
    clean_acc = accuracy(fmodel, images, labels)       # 正确率计算
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    # apply the attack
    attack = LinfPGD(steps=5)
    epsilons = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)   # epsilons 扰动量的上下界
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = 1 - success.float().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # we can also manually check this
    # we will use the clipped advs instead of the raw advs, otherwise
    # we would need to check if the perturbation sizes are actually
    # within the specified epsilon bound
    print()
    print("we can also manually check this:")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        #perturbation_sizes = (advs_ - images).norms.Linf(axis=(1, 2, 3)).numpy()
        #perturbation_sizes = torch.norm((advs_ - images),dim=0, keepdim= False).cpu().numpy()
        perturbation_sizes = torch.norm_except_dim((advs_ - images), dim=0).cpu().numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


if __name__ == "__main__":
    main()
