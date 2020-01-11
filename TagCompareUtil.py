from torch.utils.tensorboard import SummaryWriter

def writeLoss(loss,iteration,experiment):
    writer = SummaryWriter('runs/'+experiment)
    writer.add_scalar(tag='_losses',scalar_value=loss,global_step=iteration)
    writer.close()

def writeAccuracy(accuracy,iteration,experiment):
    writer = SummaryWriter('runs/'+experiment)
    writer.add_scalar(tag='_accuracies',scalar_value=accuracy,global_step=iteration)
    writer.close()