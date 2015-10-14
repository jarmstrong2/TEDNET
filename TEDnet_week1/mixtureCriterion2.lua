require 'nn'
--require 'distributions'

local MixtureCriterion, parent = torch.class('nn.MixtureCriterion', 'nn.Criterion')

-- return multivariate gauss multiplied by their respective mixture 
-- probabilities
function MixtureCriterion:getMixMultVarGauss(sigma_t, mu_t, pi_t, xTarget, batchsize)
    batchSize = xTarget:size(1)
    local sigmaTensor = sigma_t:clone():resize(batchSize, self.sizeMixture, self.dimInput)

    -- in order to perform inverse but with values on diagonal that might be zero
    sigmaTensor:add(1e-10)

    -- setting up terms for multivariate gaussian
    local sigmaTensorInverse = torch.pow(sigmaTensor, -1)
    local sigmaDetermiant = (torch.cumprod(sigmaTensor, 3)[{{},{},{self.dimInput}}]):squeeze(3)
    local muResized = mu_t:clone():resize(batchSize, self.sizeMixture, self.dimInput)
    local xTargetResized = xTarget:clone():resize(batchSize, 1, self.dimInput)
    local xTagetExpanded = xTargetResized:expand(batchSize, self.sizeMixture, self.dimInput)
    local xMinusMu = xTagetExpanded - muResized

    -- first term 1/sqrt(2pi*det(sigma))
    local term1 = torch.mul(sigmaDetermiant, (2*math.pi)^self.dimInput):sqrt():pow(-1)

    -- second term inv(sigma)*(x - mu) element-wise mult
    local term2 = torch.cmul(sigmaTensorInverse, xMinusMu)

    -- third term exp(transpose(x - mu)*term2)
    local term3 = torch.exp(torch.sum(torch.cmul(xMinusMu, term2):mul(-0.5), 3):squeeze(3))

    -- fourth term term1*term4 element-wise mult
    local term4 = torch.cmul(term1, term3)

    -- fifth term pi*term4 element-wise mult
    local term5 = torch.cmul(term4, pi_t)
    
    return term5
end

function MixtureCriterion:__init(dimInput, numMixture, isCovarianceFull)
   parent.__init(self)
   self.dimInput = dimInput
   self.sizeMixture = numMixture
   self.sizeMeanInput = dimInput * numMixture

   -- if flag isCovarianceFull true then input represents fill covariance
   if isCovarianceFull then
        self.sizeCovarianceInput = (((dimInput)*(dimInput+1))/2) * numMixture
   
   -- otherwise the input represents the main axis of a diagonal covariance
   else
        self.sizeCovarianceInput = dimInput * numMixture
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
    local piEnd = self.sizeMixture
    local pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    -- Produce a full covariance matrix from values in sigma_t
    if isCovarianceFull then
        --TODO add batchSize to CMD
        local sigmaTensor = torch.zeros(batchSize, self.sizeMixture, self.dimInput, self.dimInput)
        for i = 1, self.dimInput do
            for j = 1, self.dimInput do
                for mixCount = 0, self.sizeMixture - 1 do
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
        local mixGauss = self:getMixMultVarGauss(sigma_t, mu_t, pi_t, xTarget, batchsize)
        
        local sumMixGauss = mixGauss:sum(2):squeeze(2)

        -- apply log to sum of mixture multivariate gaussian
        local logSumGauss = torch.log(sumMixGauss)

        -- the loss function result
        lossOutput = torch.mul(logSumGauss, -1):sum() 

        --lossOutput:cmul(self.mask)

        if self.sizeAverage then
            lossOutput = lossOutput/batchSize
        end
    end
    return lossOutput
end

function MixtureCriterion:updateGradInput(input, target)
    xTarget = target:clone()
    batchSize = xTarget:size(1)

    local piStart = 1
    local piEnd = self.sizeMixture
    local pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local sigma_t = input[{{},{sigmaStart,sigmaEnd}}]
    
    if isCovarianceFull then
        
    else
        -- COMPUTE GAMMA
        -- get mixture multivariate gaussian distributions on target values
        -- multiplied by respective mixture components
        local gammaHat = self:getMixMultVarGauss(sigma_t, mu_t, pi_t, xTarget, batchsize)
        
        local sumGammaHat = torch.sum(gammaHat, 2)
    
        -- expand to size of matrix gammaHat in order to compute gamma components
        -- for each entry
        local sumGammaHatExpanded = sumGammaHat:expand(batchSize, self.sizeMixture)
    
        local gamma = torch.cmul(gammaHat, torch.pow(sumGammaHatExpanded, -1))
    
        -- TERMS FOR DERIVATIVES
        local gammaResized = gamma:clone():resize(batchSize, self.sizeMixture, 1)
        local gammaExpanded = gammaResized:expand(batchSize, self.sizeMixture, self.dimInput)
        local sigmaTensor = sigma_t:clone():resize(batchSize, self.sizeMixture, self.dimInput)
        sigmaTensor:add(1e-10)
        local sigmaTensorInverse = torch.pow(sigmaTensor, -1)
        local muResized = mu_t:clone():resize(batchSize, self.sizeMixture, self.dimInput)
        local xTargetResized = xTarget:clone():resize(batchSize, 1, self.dimInput)
        local xTagetExpanded = xTargetResized:expand(batchSize, self.sizeMixture, self.dimInput)
        local xMinusMu = xTagetExpanded - muResized
    
        -- in order to perform inverse but with values on diagonal that might be zero
        sigmaTensor:add(1e-10)

        -- setting up terms for multivariate gaussian
        local sigmaTensorInverse = torch.pow(sigmaTensor, -1)
        
        -- COMPUTE dL(x)/d(pi_t_hat)
        local d_pi_t_hat = torch.cmul(gamma, (torch.add(pi_t, -1)))
    
        -- COMPUTE dL(x)/d(mu_t_hat)
        local dl_mu_t_hat = torch.cmul(xMinusMu, sigmaTensorInverse)
        dl_mu_t_hat = torch.cmul(dl_mu_t_hat, gammaExpanded)
        dl_mu_t_hat = torch.mul(dl_mu_t_hat, -1):resize(batchSize, self.sizeMixture * self.dimInput)
    
        -- COMPUTE dL(x)/d(sigma_t_hat)
        local dl_sigma_t_hat = torch.cmul(xMinusMu, sigmaTensorInverse)
        dl_sigma_t_hat = torch.pow(dl_sigma_t_hat, 2)
        dl_sigma_t_hat = dl_sigma_t_hat - sigmaTensorInverse
        dl_sigma_t_hat = torch.mul(dl_sigma_t_hat, -0.5)
        dl_sigma_t_hat = torch.cmul(dl_sigma_t_hat, gammaExpanded)
        dl_sigma_t_hat:resize(batchSize, self.sizeMixture * self.dimInput)
    
        local grad_input = torch.cat(d_pi_t_hat:float(), dl_mu_t_hat:float())
        grad_input = torch.cat(grad_input, dl_sigma_t_hat:float())

        -- TODO add back in cuda
        --self.gradInput = grad_input:cuda()
        --self.gradInput:cmul(self.mask:reshape(self.mask:size(1),1):expand(self.gradInput:size()))
    
        if self.sizeAverage then
            self.gradInput:div(self.gradInput:size(1))
        end
        
        print(self.gradInput)
        
        return self.gradInput
    end
end
