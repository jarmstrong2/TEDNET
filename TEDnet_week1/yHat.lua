require 'nn'

local YHat, parent = torch.class('nn.YHat', 'nn.Module')

function YHat:__init(dimInput, numMixture, isCovarianceFull)
   parent.__init(self)
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

function YHat:updateOutput(input)
    local piStart = 1
    local piEnd = self.sizeMixture
    local hat_pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local hat_mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local hat_sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    --self.pi_t_act = self.pi_t_act or nn.SoftMax():cuda()
   self.pi_t_act = self.pi_t_act or nn.SoftMax()
   
    local pi_t = self.pi_t_act:forward(hat_pi_t)
    local mu_t = hat_mu_t:clone()
    local sigma_t = hat_sigma_t:clone()
    
    local output = torch.cat(pi_t:float(), mu_t:float(), 2)
    output = torch.cat(output, sigma_t:float(), 2)

    self.output = output
    
    return self.output
end

function YHat:updateGradInput(input, gradOutput)
    local piStart = 1
    local piEnd = self.sizeMixture
    local hat_pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local hat_mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local hat_sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    local d_hat_pi_t_softmax = self.pi_t_act:backward(hat_pi_t, 
       gradOutput[{{},{piStart,piEnd}}])
    --local d_hat_pi_t = gradOutput[{{},{piStart,piEnd}}]
    local d_hat_mu_t = gradOutput[{{},{muStart,muEnd}}]
    local d_hat_sigma_t = gradOutput[{{},{sigmaStart,sigmaEnd}}]

    local grad_hat_pi_t = torch.cmul(d_hat_pi_t, d_hat_pi_t_softmax)
    local grad_hat_mu_t = d_hat_mu_t:clone()
    local grad_hat_sigma_t = d_hat_sigma_t:clone()
        
    local grad_input = torch.cat(grad_hat_pi_t:float(), grad_hat_mu_t:float(), 2)
    grad_input = torch.cat(grad_input, grad_hat_sigma_t:float(), 2)
    
    self.gradInput = grad_input
    
    return self.gradInput  
end
