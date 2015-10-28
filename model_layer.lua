require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'
require 'nngraph'
require 'optim'
require 'getvocab'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
require 'window'
require 'yHat'
require 'mixtureCriterion'
local model_utils=require 'model_utils'
require 'cunn'
require 'distributions'
torch.manualSeed(123)

print('start model making')

-- make model
model = {}

model.criterion = nn.MixtureCriterion():cuda()
model.criterion:setSizeAverage()

local input_xin = nn.Identity()()
local input_context = nn.Identity()()
local input_w_prev = nn.Identity()()
local input_lstm_h1_h = nn.Identity()()
local input_lstm_h1_c = nn.Identity()()
local input_lstm_h2_h = nn.Identity()()
local input_lstm_h2_c = nn.Identity()()
local input_lstm_h3_h = nn.Identity()()
local input_lstm_h3_c = nn.Identity()()
local input_prev_kappa = nn.Identity()()

local h1 = LSTMH1.lstm(opt.inputSize, opt.hiddenSize)({input_xin, input_w_prev, input_lstm_h1_c, input_lstm_h1_h})
local h1_c = nn.SelectTable(1)(h1)
local h1_h = nn.SelectTable(2)(h1)
local w_output = nn.Window()({nn.Linear(opt.hiddenSize,30)(h1_h), input_context, input_prev_kappa})
local w_vector = nn.SelectTable(1)(w_output)
local w_kappas_t = nn.SelectTable(2)(w_output)
local w_phi_t = nn.SelectTable(3)(w_output)
local h2 = LSTMHN.lstm(opt.inputSize, opt.hiddenSize)({input_xin, w_vector, h1_h, input_lstm_h2_c, input_lstm_h2_h})
local h2_c = nn.SelectTable(1)(h2)
local h2_h = nn.SelectTable(2)(h2)
local h3 = LSTMHN.lstm(opt.inputSize, opt.hiddenSize)({input_xin, w_vector, h2_h, input_lstm_h3_c, input_lstm_h3_h})
local h3_c = nn.SelectTable(1)(h3)
local h3_h = nn.SelectTable(2)(h3)
--if opt.isCovarianceFull then
--else
	local y = nn.YHat()(nn.Linear(opt.hiddenSize*3, (opt.numMixture + 2 * (opt.inputSize * opt.numMixture)))
		(nn.JoinTable(2)({h1_h, h2_h, h3_h})))
--end

model.rnn_core = nn.gModule({input_xin, input_context, input_prev_kappa, input_w_prev,  
                             input_lstm_h1_c, input_lstm_h1_h,
                             input_lstm_h2_c, input_lstm_h2_h,
                             input_lstm_h3_c, input_lstm_h3_h},
                            {y, w_kappas_t, w_vector, w_phi_t, h1_c, h1_h, h2_c, h2_h,
                             h3_c, h3_h})

model.rnn_core:cuda()
params, grad_params = model.rnn_core:getParameters()

-- make a bunch of clones, AFTER flattening, as that reallocates memory
MAXLEN = opt.maxlen
clones = {} -- TODO: local
for name,mod in pairs(model) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times_fast(mod, MAXLEN-1, not mod.parameters)
end

print('end model making')
