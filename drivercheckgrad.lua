local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for checkgrad.')

cmd:option('-inputSize' , 61, 'number of input dimension')
cmd:option('-hiddenSize' , 400, 'number of hidden units in lstms')
cmd:option('-lr' , 1e-4, 'learning rate')
cmd:option('-maxlen' , 1, 'max sequence length')
cmd:option('-batchSize' , 4, 'mini batch size')
cmd:option('-numPasses' , 1, 'number of passes')
cmd:option('-isCovarianceFull' , false, 'true if full covariance, o.w. diagonal covariance')
cmd:option('-numMixture' , 10, 'number of mixture components in output layer')

cmd:text()
opt = cmd:parse(arg)

dofile('model.lua')
dofile('checkgrad.lua')
