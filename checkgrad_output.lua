require 'yHat'
require 'mixtureCriterion'
require 'optim'
require 'torch'
require 'nn'
require 'cunn'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for training model.')

cmd:option('-inputSize' , 30, 'number of input dimension')
cmd:option('-hiddenSize' , 400, 'number of hidden units in lstms')
cmd:option('-lr' , 1e-5, 'learning rate')
cmd:option('-maxlen' , 2, 'max sequence length')
cmd:option('-batchSize' , 4, 'mini batch size')
cmd:option('-numPasses' , 4, 'number of passes')
cmd:option('-isCovarianceFull' , false, 'true if full covariance, o.w. diagonal covariance')
cmd:option('-numMixture' , 1, 'number of mixture components in output layer') 

cmd:text()
opt = cmd:parse(arg)

y_size = opt.numMixture + 2 * (opt.inputSize * opt.numMixture)

l = nn.Linear(5, y_size):cuda()
y = nn.YHat():cuda()

s = nn.Sequential():cuda()
s:add(l)
s:add(y)

mixture = nn.MixtureCriterion():cuda()
mixture:setSizeAverage()

params, grad_params = s:getParameters()

print(params:size())

mask = torch.ones(2, 1):cuda()
mixture:setmask(mask)

input = torch.CudaTensor(2, 5)
input:uniform(-0.08, 0.08)

target = torch.randn(2, opt.inputSize):cuda()

function feval(x)
	if x ~= params then
        	params:copy(x)
    	end
    	grad_params:zero()

	output = s:forward(input)
	loss = mixture:forward(output, target)
	mixgrad = mixture:backward(output, target)
	grad_y = s:backward(input, mixgrad)

	return loss, grad_params:double() 	
end

diff, dC, dC_est = optim.checkgrad(feval, params, 1e-2)

