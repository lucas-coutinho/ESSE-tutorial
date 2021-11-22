import argparse
import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
import time


try:
    import pandas as pd 
    has_pandas = True
except:
    has_pandas = False
import matplotlib.pyplot as plt

def run_benchmark(model_file, img_loader, num_batches):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    # Run the scripted model on a few batches of images
    with torch.no_grad():
        for i, (images, target) in enumerate(img_loader):
            if i < num_batches:
                start = time.time()
                output = model(images)
                end = time.time()
                elapsed = elapsed + (end-start)
            else:
                break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.3f ms' % (elapsed/num_images*1000))
    return elapsed

def main():

    random_images_for_benchmarking = torch.Tensor(100,3,32,32) #100 random images
    img_loader = DataLoader(list(zip(random_images_for_benchmarking, torch.Tensor(100,1))), batch_size=1)
    torch.backends.quantized.engine = 'qnnpack'
               
    time_elapses = pd.DataFrame([]) if has_pandas else {}
    if args.path_to_float_model:
        print("Full precision Model:")
        time_elapses['FullPrecision'] = [run_benchmark(args.path_to_float_model, img_loader,1) for i in range(10)]

    if args.path_to_postquant:
        print("Post training quantization:")
        time_elapses['PostQ'] = [run_benchmark(args.path_to_postquant, img_loader,1) for _ in range(10)]
    
    if args.path_to_qat:
        print("Quantization-Aware Model:")
        time_elapses['QAT'] = [run_benchmark(args.path_to_qat, img_loader,1) for _ in range(10)]
        

    print('\n\n\n')

    if has_pandas:
        print(time_elapses.mean(axis=0)*1000)
    else:
        for key, value in time_elapses.items():
            print(key+": ", sum(value)/len(value))
    
    

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--path-to-float-model', type=str, help='Path where is stored the full precision model', default=None)
        parser.add_argument('--path-to-postquant', type=str, help='Path where is stored the Post training quantized model', default=None)
        parser.add_argument('--path-to-qat', type=str, help='Path where is stored the Quantization Aware quantized model', default=None)
        parser.add_argument('--log', type=str, help='log file name', default='stat.log')
        args = parser.parse_args()

        main()