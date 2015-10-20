require 'nn'
require 'cunn' 

local MixtureCriterion, parent = torch.class('nn.MixtureCriterion', 'nn.Criterion')

-- return multivariate gauss multiplied by their respective mixture 
-- probabilities
function MixtureCriterion:getMixMultVarGauss(sigma_t, mu_t, pi_t, xTarget, batchsize)
    batchSize = xTarget:size(1)
    local sigmaTensor = sigma_t:clone():resize(batchSize, opt.numMixture, opt.inputSize)

    -- in order to perform inverse but with values on diagonal that might be zero
    sigmaTensor:add(1e-10)

    -- setting up terms for multivariate gaussian
    local sigmaTensorInverse = torch.pow(sigmaTensor, -1):cuda()
    
    local sigmaDetermiant = (torch.cumprod(sigmaTensor, 3)[{{},{},{opt.inputSize}}]):squeeze(3):cuda()
    local muResized = mu_t:clone():resize(batchSize, opt.numMixture, opt.inputSize):cuda()
    local xTargetResized = xTarget:clone():resize(batchSize, 1, opt.inputSize):cuda()
    local xTagetExpanded = xTargetResized:expand(batchSize, opt.numMixture, opt.inputSize):cuda()
    local xMinusMu = xTagetExpanded:cuda() - muResized

    -- first term 1/sqrt(2pi*det(sigma))
    local term1 = ((torch.mul(sigmaDetermiant, (2*math.pi)^(opt.inputSize)):sqrt()):pow(-1)) --* ((2*math.pi)^(-opt.inputSize/2))

    -- second term inv(sigma)*(x - mu) element-wise mult
    local term2 = torch.cmul(sigmaTensorInverse, xMinusMu)

    -- third term exp(transpose(x - mu)*term2)
    --local term3 = torch.exp(torch.sum(torch.cmul(xMinusMu, term2):mul(-0.5), 3):squeeze(3):clamp(-(1/0),80))
    local term3 = torch.exp(torch.sum(torch.cmul(xMinusMu, term2):mul(-0.5), 3):squeeze(3))
        
    -- fourth term term1*term3 element-wise mult
    local term4 = torch.cmul(term1, term3)

    -- fifth term pi*term4 element-wise mult
    local term5 = torch.cmul(term4, pi_t:cuda())
     
	--print(term5)
   
    return term5
end

function MixtureCriterion:__init()
   parent.__init(self)
   self.sizeMeanInput = opt.inputSize * opt.numMixture

   -- if flag opt.isCovarianceFull true then input represents fill covariance
   if opt.isCovarianceFull then
        self.sizeCovarianceInput = (((opt.inputSize)*(opt.inputSize+1))/2) * opt.numMixture
   
   -- otherwise the input represents the main axis of a diagonal covariance
   else
        self.sizeCovarianceInput = opt.inputSize * opt.numMixture
   end
end

function MixtureCriterion:setmask(mask)
   self.mask = mask 
end

function MixtureCriterion:setSizeAverage()
   self.sizeAverage = true 
end

function MixtureCriterion:updateOutput(input, target)
    xTarget = target:clone()
    batchSize = xTarget:size(1)

    local piStart = 1
    local piEnd = opt.numMixture
    local pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    -- Produce a full covariance matrix from values in sigma_t
    if opt.isCovarianceFull then
        --TODO add batchSize to CMD
        local sigmaTensor = torch.zeros(batchSize, opt.numMixture, opt.inputSize, opt.inputSize)
        for i = 1, opt.inputSize do
            for j = 1, opt.inputSize do
                for mixCount = 0, opt.numMixture - 1 do
                    if j <= i then
                        sigmaPoint = sigma_t[{{},{(i * (i-1))/2 + j + mixCount}}]
                    else
                        sigmaPoint = sigma_t[{{},{(j * (j-1))/2 + i + mixCount}}]
                    end
                    sigmaTensor[{{}, {mixCount + 1}, {i}, {j}}] = sigmaPoint:clone()
                end
            end
        end

    -- Produce a diagonal matrix represented by a vector from values in sigma_t
    else
        -- get mixture multivariate gaussian distributions on target values
        -- multiplied by respective mixture components
        --local mixGauss = self:getMixMultVarGauss(sigma_t:double(), mu_t:double(), pi_t:double(), xTarget:double(), batchsize)
        local mixGauss = self:getMixMultVarGauss(sigma_t, mu_t, pi_t, xTarget, batchsize)
 --	print(mixGauss)       
        local sumMixGauss = mixGauss:sum(2):squeeze(2)

        -- apply log to sum of mixture multivariate gaussian
        local logSumGauss = torch.log(sumMixGauss)
        
        -- the loss function result
        lossOutput = torch.mul(logSumGauss, -1) 

        lossOutput = lossOutput:cmul(self.mask):sum()

        if self.sizeAverage then
            lossOutput = lossOutput/batchSize
        end
    end
   -- print(lossOutput)
    return lossOutput
end

function MixtureCriterion:updateGradInput(input, target)
    xTarget = target:clone()
    batchSize = xTarget:size(1)

    local piStart = 1
    local piEnd = opt.numMixture
    local pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local sigma_t = input[{{},{sigmaStart,sigmaEnd}}]
    
    if opt.isCovarianceFull then
        
    else
        -- COMPUTE GAMMA
        -- get mixture multivariate gaussian distributions on target values
        -- multiplied by respective mixture components
        --local gammaHat = self:getMixMultVarGauss(sigma_t:double(), mu_t:double(), pi_t:double(), xTarget:double(), batchsize)
        local gammaHat = self:getMixMultVarGauss(sigma_t, mu_t, pi_t, xTarget, batchsize)
        
        local sumGammaHat = torch.sum(gammaHat, 2)
    
        -- expand to size of matrix gammaHat in order to compute gamma components
        -- for each entry
        local sumGammaHatExpanded = sumGammaHat:expand(batchSize, opt.numMixture)
    
        local gamma = torch.cmul(gammaHat, torch.pow(sumGammaHatExpanded + 1e-10, -1))
    
        -- TERMS FOR DERIVATIVES
        local gammaResized = gamma:clone():resize(batchSize, opt.numMixture, 1):cuda()
        local gammaExpanded = gammaResized:expand(batchSize, opt.numMixture, opt.inputSize):cuda()
        local sigmaTensor = sigma_t:clone():resize(batchSize, opt.numMixture, opt.inputSize):cuda()
        local sigmaTensorInverse = torch.pow(sigmaTensor, -1):cuda()
        local muResized = mu_t:clone():resize(batchSize, opt.numMixture, opt.inputSize):cuda()
        local xTargetResized = xTarget:clone():resize(batchSize, 1, opt.inputSize):cuda()
        local xTagetExpanded = xTargetResized:expand(batchSize, opt.numMixture, opt.inputSize):cuda()
        local xMinusMu = xTagetExpanded:cuda() - muResized
    
        -- setting up terms for multivariate gaussian
        local sigmaTensorInverse = torch.pow(sigmaTensor, -1):cuda()
        
        -- COMPUTE dL(x)/d(pi_t_hat)
        --local d_pi_t_hat = torch.cmul(gamma, (torch.add(pi_t, -1)))
        --local d_pi_t_hat = torch.mul(torch.cmul(gamma, (torch.pow(torch.add(pi_t, 1e-10), -1))), -1)
        local d_pi_t_hat = pi_t:cuda() - gamma
    
        -- COMPUTE dL(x)/d(mu_t_hat)
        local dl_mu_t_hat = torch.cmul(xMinusMu, sigmaTensorInverse)
        dl_mu_t_hat = torch.cmul(dl_mu_t_hat, gammaExpanded)
        dl_mu_t_hat = torch.mul(dl_mu_t_hat, -1):resize(batchSize, opt.numMixture * opt.inputSize)
    
        -- COMPUTE dL(x)/d(sigma_t_hat)
        local dl_sigma_t_hat = torch.cmul(xMinusMu, sigmaTensorInverse)
        dl_sigma_t_hat = torch.pow(dl_sigma_t_hat, 2)
        dl_sigma_t_hat = dl_sigma_t_hat - sigmaTensorInverse
        dl_sigma_t_hat = torch.mul(dl_sigma_t_hat, -0.5)
        dl_sigma_t_hat = torch.cmul(dl_sigma_t_hat, gammaExpanded)
        dl_sigma_t_hat:resize(batchSize, opt.numMixture * opt.inputSize)
    
        local grad_input = torch.cat(d_pi_t_hat:float(), dl_mu_t_hat:float())
        grad_input = torch.cat(grad_input, dl_sigma_t_hat:float())
--	print("gin")
--	print(grad_input)

        self.gradInput = grad_input:cuda()
        self.gradInput:cmul(self.mask:reshape(self.mask:size(1),1):expand(self.gradInput:size()))
    
        if self.sizeAverage then
            self.gradInput:div(self.gradInput:size(1))
        end
        
        return self.gradInput
    end
end
