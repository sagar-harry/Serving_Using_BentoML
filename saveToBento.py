from trained_model import mnist_pred
from mnist import numberClassifier
import utils1
import argparse

def saveToBento(checkpoint):
    model_state_dict, _, _, _ = utils1.load_model(checkpoint)
    model_ft = mnist_pred()
    model_ft.load_state_dict(model_state_dict)
    bento_svc = numberClassifier()
    bento_svc.pack("model", model_ft)
    saved_path = bento_svc.save()
    print('Bento Service Saved in ', saved_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint to load the model')
    args = parser.parse_args()
    saveToBento(args.checkpoint)