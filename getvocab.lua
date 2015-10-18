require 'torch'

--get vocab
count = 0
vocabstr = "abcdefghijklmnopqrstuvwxyz,.!? 0"
vocab = {}
for i = 1, #vocabstr do
    count = count + 1
    vocab[vocabstr:sub(i,i)] = count 
end

cuArray = torch.eye(32)

function getOneHotChar(c)
    local index = vocab[c]
    local oneHotChar = cuArray[index]
    return oneHotChar
end

function getOneHotStr(s)
    local oneHotStr = nil
    for c in s:gmatch"." do
        
        res = c:gsub('[^a-z ,.!?]','')
        if res ~= '' then
            inputchar = c
        else
            inputchar = '0'
        end

        if not oneHotStr then
            oneHotStr = getOneHotChar(inputchar):float()
        else
            oneHotStr = torch.cat(oneHotStr:float(), getOneHotChar(inputchar):float(), 2)
        end
    end
    return oneHotStr:clone()
end

function getOneHotStrs(strs)

-- will be given as an array of strs to be converted into one 
-- hot array of arrays

    maxCharLen = 0

    for i = 1, #strs do
        charLen = #strs[i]
        if charLen > maxCharLen then
            maxCharLen = charLen
        end 
    end
    
    --allOneHot = torch.zeros(83, maxCharLen, #strs)
    allOneHot = torch.zeros(#strs, maxCharLen, 32)
    
    for i = 1, #strs do
        strLen = #(strs[i])
        charRemain = maxCharLen - strLen
        oneHot = getOneHotStr(strs[i])
        if charRemain > 0 then
            zeroOneHotVectors = torch.zeros(32, charRemain)
            finalOneHot = torch.cat(oneHot:float(), zeroOneHotVectors:float(),2)
            allOneHot[{{i},{},{}}] = finalOneHot:t()
        else
            allOneHot[{{i},{},{}}] = oneHot:t()
        end
    end 

    return allOneHot
end
