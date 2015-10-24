require 'getbatch'
require 'gnuplot'

-- get training dataset
STRAIGHTdata = torch.load('/home/jarmstrong/TEDNET/toy_600.t7')
dataSize = #STRAIGHTdata

print('uploaded training data')

-- get validation dataset
valSTRAIGHTdata = torch.load('/home/jarmstrong/TEDNET/toy_600.t7')
valdataSize = #valSTRAIGHTdata

print('uploaded validation data')

print('start training')

params:uniform(-0.008, 0.008)
sampleSize = opt.batchSize
numberOfPasses = opt.numPasses

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(sampleSize, opt.hiddenSize):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()
initstate_h3_c = initstate_h1_c:clone()
initstate_h3_h = initstate_h1_c:clone()
initstate_h4_c = initstate_h1_c:clone()
initstate_h4_h = initstate_h1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
dfinalstate_h1_c = initstate_h1_c:clone()
dfinalstate_h1_h = initstate_h1_c:clone()
dfinalstate_h2_c = initstate_h1_c:clone()
dfinalstate_h2_h = initstate_h1_c:clone()
dfinalstate_h3_c = initstate_h1_c:clone()
dfinalstate_h3_h = initstate_h1_c:clone()
dfinalstate_h4_c = initstate_h1_c:clone()
dfinalstate_h4_h = initstate_h1_c:clone()
initkappa = torch.randn(sampleSize,10)
dinitkappa = torch.zeros(sampleSize,10)

count = 1

batchCount = nil

function getInitW(cuMat)
	cuMatClone = cuMat:clone()
    return torch.zero(cuMatClone[{{},{1},{}}]:squeeze(2))
end

function getValLoss()
    local valnumberOfPasses = opt.numPasses
    local valcount = 550
    local valsampleSize = opt.batchSize
    local loss = 0
    local elems = 0
    
    -- add for loop to increase mini-batch size
    for i=1, valnumberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, cuMat, cmaskMat, elementCount, valcount
        = getBatch(valcount, valSTRAIGHTdata, valsampleSize)
        ------------------------------------------------------------

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local w = {[0]=getInitW(cuMat:cuda())}

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        local lstm_c_h4 = {[0]=initstate_h4_c} -- internal cell states of LSTM
        local lstm_h_h4 = {[0]=initstate_h4_h} -- output values of LSTM        

        local kappa_prev = {[0]=torch.zeros(sampleSize,10):cuda()}
        
        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        
        -- forward
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{1,opt.inputSize},{t}}]:squeeze(3)
            local x_target = inputMat[{{},{1,opt.inputSize},{t+1}}]:squeeze(3)

            -- model 
            output_y[t], kappa_prev[t], w[t], _, lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], lstm_c_h4[t], lstm_h_h4[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]}))
       
            -- criterion 
            clones.criterion[t]:setmask(cmaskMat[{{},{},{t}}]:cuda())
            loss = clones.criterion[t]:forward(output_y[t], x_target:cuda()) + loss        
        end

        maxLen = nil
        strs = nil
        inputMat = nil 
        maskMat = nil
        cuMat = nil
        w = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        kappa_prev = nil
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    local loss = 0
    local elems = 0
    
    -- add for loop to increase mini-batch size
    for i=1, numberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, cuMat, cmaskMat, elementCount, count
        = getBatch(count, STRAIGHTdata, sampleSize)
        ------------------------------------------------------------

if count > 500 then
count = 1
end

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local w = {[0]=getInitW(cuMat:cuda())}

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        local lstm_c_h4 = {[0]=initstate_h4_c} -- internal cell states of LSTM
        local lstm_h_h4 = {[0]=initstate_h4_h} -- output values of LSTM
        
        local kappa_prev = {[0]=torch.zeros(sampleSize,10):cuda()}
        
        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        
        -- FORWARD
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{1,opt.inputSize},{t}}]:squeeze(3)
            local x_target = inputMat[{{},{1,opt.inputSize},{t+1}}]:squeeze(3)

            -- model 
            output_y[t], kappa_prev[t], w[t], _, lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], lstm_c_h4[t], lstm_h_h4[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]}))
       
            -- criterion 
            clones.criterion[t]:setmask(cmaskMat[{{},{},{t}}]:cuda())
            --print(output_y[t])
            --print(x_target)
	    loss = clones.criterion[t]:forward(output_y[t], x_target:cuda()) + loss
            --print('inner loop ',loss)        
        end
        --print('current pass ',loss)        
        elems = (elementCount - sampleSize) + elems
        
        -- BACKWARD
                
        local dlstm_c_h1 = dfinalstate_h1_c
        local dlstm_h_h1 = dfinalstate_h1_h
        local dlstm_c_h2 = dfinalstate_h2_c
        local dlstm_h_h2 = dfinalstate_h2_h
        local dlstm_c_h3 = dfinalstate_h3_c
        local dlstm_h_h3 = dfinalstate_h3_h
        local dlstm_c_h4 = dfinalstate_h4_c
        local dlstm_h_h4 = dfinalstate_h4_h
        
        local dh1_w = torch.zeros(sampleSize, 32):cuda()
        local dkappa = torch.zeros(sampleSize, 10):cuda()
        
        for t = maxLen - 1, 1, -1 do
        
            local x_in = inputMat[{{},{1,opt.inputSize},{t}}]:squeeze()
            local x_target = inputMat[{{},{1,opt.inputSize},{t+1}}]:squeeze()
            
            -- criterion
            local grad_crit = clones.criterion[t]:backward(output_y[t]:cuda(), x_target:cuda())            
            	
            -- model
            _x, _c, dkappa, dh1_w, dlstm_c_h1, dlstm_h_h1,
            dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3 = unpack(clones.rnn_core[t]:backward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]},
                 {grad_crit:cuda(), dkappa, dh1_w, _, dlstm_c_h1, dlstm_h_h1, 
                  dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3, dlstm_c_h4, dlstm_h_h4}))
        end

        dh2_w = nil
        dh2_h1 = nil
        dh3_w = nil
        dh3_h2 = nil
        maxLen = nil
        strs = nil
        inputMat = nil 
        maskMat = nil
        cuMat = nil
        w = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        dlstm_c_h1 = nil -- internal cell states of LSTM
        dlstm_h_h1 = nil -- internal cell states of LSTM
        dlstm_c_h2 = nil -- internal cell states of LSTM
        dlstm_h_h2 = nil -- internal cell states of LSTM
        dlstm_c_h3 = nil -- internal cell states of LSTM
        dlstm_h_h3 = nil -- internal cell states of LSTM
        dkappaNext = nil
        dh1_w_next = nil
        kappa_prev = nil
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    
    grad_params:div(numberOfPasses)
    
    -- clip gradient element-wise
    grad_params:clamp(-10, 10)
     
    return loss, grad_params
end

vallosses = nil
losses = nil 
local optim_state = {learningRate = opt.lr, alpha = 0.95, epsilon = 1e-8}
local iterations = 800000
local minValLoss = 1/0
for i = 1, iterations do
    batchCount = i

    local _, loss = optim.adam(feval, params, optim_state)

    print(string.format("update param, loss = %6.8f, gradnorm = %6.4e", loss[1], grad_params:clone():norm()))
    if i % 100 == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
        valLoss = getValLoss()

        print(string.format("validation loss = %6.8f", valLoss))
        if minValLoss > valLoss then
            minValLoss = valLoss
            torch.save("tednet2.t7", model)
            print("------- Model Saved --------")
        end
        
        if valLoss ~= 1/0 then
		if not vallosses then
		 	vallosses = torch.Tensor(1)
		 	vallosses[1] = valLoss
			valiter = torch.Tensor(1)
			valiter[1] = i
		else
			vallossesaddition = torch.Tensor(1)
        	        vallossesaddition[1] = valLoss
        	        valiteraddition = torch.Tensor(1)
        	        valiteraddition[1] = i
			vallosses = torch.cat(vallosses:float(), vallossesaddition:float(), 1)
			valiter = torch.cat(valiter, valiteraddition, 1)
		end
        end
        
        if loss[1] ~= 1/0 then
		if not losses then
        	        losses = torch.Tensor(1)
         		losses[1] = loss[1]
                	iter = torch.Tensor(1)
                	iter[1] = i
        	else
                	lossesaddition = torch.Tensor(1,1)
                	lossesaddition[1] = loss[1]
                	iteraddition = torch.Tensor(1)
                	iteraddition[1] = i
                	losses = torch.cat(losses:float(), lossesaddition:float(),1)
                	iter = torch.cat(iter, iteraddition, 1)
        	end
        end
	gnuplot.pngfigure(opt.lossImageFN)
	gnuplot.plot({iter, losses},{valiter, vallosses})
	gnuplot.plotflush()
    end
end
