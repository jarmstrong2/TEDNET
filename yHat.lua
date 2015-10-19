require 'nn'

local YHat, parent = torch.class('nn.YHat', 'nn.Module')

function YHat:__init()
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

function YHat:updateOutput(input)
    local piStart = 1
    local piEnd = opt.numMixture
    local hat_pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local hat_mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local hat_sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    self.pi_t_act = self.pi_t_act or nn.SoftMax():cuda()
    self.sigma_t_act = self.sigma_t_act or nn.Exp():cuda()
   
    local pi_t = self.pi_t_act:forward(hat_pi_t)
    local mu_t = hat_mu_t:clone()
    local sigma_t = self.sigma_t_act:forward(hat_sigma_t)
    
    local output = torch.cat(pi_t:float(), mu_t:float(), 2)
    output = torch.cat(output, sigma_t:float(), 2)

    self.output = output
    
    return self.output
end

function YHat:updateGradInput(input, gradOutput)
    local piStart = 1
    local piEnd = opt.numMixture
    local hat_pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local hat_mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local hat_sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    local d_hat_pi_t = gradOutput[{{},{piStart,piEnd}}]
    local d_hat_mu_t = gradOutput[{{},{muStart,muEnd}}]
    local d_hat_sigma_t = gradOutput[{{},{sigmaStart,sigmaEnd}}]

    local grad_hat_pi_t = d_hat_pi_t:clone()
    local grad_hat_mu_t = d_hat_mu_t:clone()
    local grad_hat_sigma_t = self.sigma_t_act:backward(hat_sigma_t,d_hat_sigma_t)
        
    local grad_input = torch.cat(grad_hat_pi_t:float(), grad_hat_mu_t:float(), 2)
    grad_input = torch.cat(grad_input, grad_hat_sigma_t:float(), 2)
    
    self.gradInput = grad_input:cuda() 

    return self.gradInput  
end

