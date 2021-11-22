# ESSE-tutorial

Aqui contém o necessário para executar o tutorial apresentado no ESSE 2021.


## **Requisitos**

```bash
torch==1.5.0
torchvision==0.5.0
```

## **Uso**

```bash
usage: benchmarking_in_rpi.py [-h] [--path-to-float-model PATH_TO_FLOAT_MODEL]
                              [--path-to-postquant PATH_TO_POSTQUANT]
                              [--path-to-qat PATH_TO_QAT] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --path-to-float-model PATH_TO_FLOAT_MODEL
                        Path where is stored the full precision model
  --path-to-postquant PATH_TO_POSTQUANT
                        Path where is stored the Post training quantized model
  --path-to-qat PATH_TO_QAT
                        Path where is stored the Quantization Aware quantized
                        model
```

## **Uso com docker**
Para rodar o jupyter notebook com docker é necessário:
-   Instalar jupyter lab
-   Instalar docker e docker-compose
-   executar ```bash source build.sh && source run.sh``` 
-   Utilizar o notebook da mesma forma q no colab
