require 'nn'
require 'distributions'

local MixtureCriterion, parent = torch.class('nn.MixtureCriterion', 'nn.Criterion')

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
    x_target = target:clone()

    local piStart = 1
    local piEnd = self.sizeMixture
    local pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    local sigmaTensor = torch.zeros(batchSize, self.sizeMixture, self.dimInput, self.dimInput)

    -- Produce a full covariance matrix from values in sigma_t
    if isCovarianceFull then
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

    -- Produce a diagonal matrix from values in sigma_t
    else
        for i = 1, self.dimInput do
            for mixCount = 0, self.sizeMixture - 1 do
                sigmaPoint = sigma_t[{{},{i + mixCount}]
                sigmaTensor[{{}, {mixCount + 1}, {i}, {i}}] = sigmaPoint:clone()
            end 
        end
    end
end

function MixtureCriterion:updateGradInput(input, target)
    
end
