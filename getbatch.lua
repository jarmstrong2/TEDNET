require 'torch'
require 'getvocab'

sample_size = 3
function getMaxLen(newLen, remainingLen, count, data)
    maxLen = 0
    for i = count, remainingLen do
        inputLen = (#(data[i].mat))[1]
        if inputLen > maxLen then
            maxLen = inputLen
        end
    end
    for i = 1, newLen do
        inputLen = (#(data[i].mat))[1]
        if inputLen > maxLen then
            maxLen = inputLen
        end
    end
    return maxLen
end

function getLens(count, data)
    dataLen = #data
    if (count + (sample_size - 1)) > dataLen then
        newLen = (count + (sample_size - 1)) - dataLen
        remainingLen = (count + (sample_size - 1)) - newLen 
    else
        newLen = 0
        remainingLen = (count + (sample_size - 1))
    end
    return newLen, remainingLen
end

function getStrs(newLen, remainingLen, count, data)
    strs = {}
    for i = count, remainingLen do
        table.insert(strs, data[i].str)
    end
    for i = 1, newLen do
        table.insert(strs, data[i].str)
    end
    return strs
end

function getInputAndMaskMat(maxLen, newLen, remainingLen, count, data)
    sampleCount = 1
    inputMat = torch.zeros(sample_size, opt.inputSize, maxLen)
    cmaskMat = torch.zeros(sample_size, 1, maxLen)
    elementCount = 0
    for i = count, remainingLen do
        for j = 1, (#(data[i].mat))[1] do
            inputMat[{{sampleCount}, {}, {j}}] = (data[i].mat[{{j},{1,opt.inputSize}}])
            cmaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            if j <= 1000 then
                elementCount = elementCount + 1
            end
        end
        sampleCount = sampleCount + 1
    end
    for i = 1, newLen do
        for j = 1, (#(data[i].mat))[1] do
            inputMat[{{sampleCount}, {}, {j}}] = (data[i].mat[{{j},{1,opt.inputSize}}])
            cmaskMat[{{sampleCount}, {}, {j}}]:fill(1)
            if j <= 1000 then
                elementCount = elementCount + 1
            end
        end
        sampleCount = sampleCount + 1
    end
    return inputMat, cmaskMat, elementCount
end

function getNewCount(newLen, remainingLen)
    if newLen == 0 then
        newCount = remainingLen + 1
    else
        newCount = newLen + 1
    end
    return newCount
end

function getBatch(count, data, sampsize)
    sample_size = sampsize
    newLen, remainingLen = getLens(count, data)
    maxLen = getMaxLen(newLen, remainingLen, count, data)
    strs = getStrs(newLen, remainingLen, count, data)
    cu = getOneHotStrs(strs)
    inputMat, cmaskMat, elementCount = getInputAndMaskMat(maxLen, newLen, remainingLen, count, data)
    newCount = getNewCount(newLen, remainingLen)
    return maxLen, strs, inputMat, cu, cmaskMat, elementCount, newCount
end
