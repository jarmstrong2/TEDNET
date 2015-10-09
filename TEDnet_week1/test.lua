require 'torch'
require 'mixtureCriterion'
y = nn.MixtureCriterion(2,3,false)
input = torch.ones(2,15)
input[{{1,2},{1}}]:fill(0.2)
input[{{1,2},{2}}]:fill(0.1)
input[{{1,2},{3}}]:fill(0.7)

input[{{1,2},{4}}]:fill(1)
input[{{1,2},{5}}]:fill(1)

input[{{1,2},{6}}]:fill(2)
input[{{1,2},{7}}]:fill(2)

input[{{1,2},{8}}]:fill(3)
input[{{1,2},{9}}]:fill(3)

input[{{1,1},{10,15}}]:fill(-1)

target = torch.Tensor(2,2):fill(2)
q = y:forward(input, target)