# SubmesoFluxNet
Machine learning for ocean submesoscale heat flux parameterization

##  run
This repo expects `X.npy` (N, C, H, W) and `y.npy` (N, 1, H, W)

where: N - sample number, C - input channel, H - images length, W - image height

loss function option:
"Huber", "MSE", "MAE" 

optim options:
"AdamW", "Adam"
  
