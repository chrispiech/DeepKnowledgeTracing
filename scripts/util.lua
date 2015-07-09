function selfDot(m)
	return torch.dot(m, m)
end

function tableToVec(gradList)
	local vec = gradList[1]
	for i = 2,#gradList do
		vec = torch.cat(vec, gradList[i], 1)
	end
	return vec;
end

function saveMatrixAsCsv(path, matrix)
	local file = io.open(path, "w");
	for r = 1, matrix:size(1) do
		for c = 1, matrix:size(2) do
			file:write(matrix[r][c])
			if(c ~= matrix:size(2)) then
				file:write(',')
			end
		end
		file:write('\n')
	end
	file:close()
end

function split(str, sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    str:gsub(pattern, function(c) fields[#fields+1] = c end)
    return fields
end

function saveMatrixAsGephi(path, matrix)
	local file = io.open(path, "w");
	-- write first line
	for c = 1, matrix:size(2) do
		file:write(';' .. c)
	end
	file:write('\n')

	for r = 1, matrix:size(1) do
		file:write(r)
		for c = 1, matrix:size(2) do
			file:write(';')
			file:write(matrix[r][c])
		end
		file:write('\n')
	end
	file:close()
end

function copyWithoutRow(m, index)
	local nRows = m:size(1)
	if(index == 1) then
		return m:narrow(1, 2, nRows - 1):clone()
	end
	if(index == nRows) then
		return m:narrow(1, 1, nRows - 1):clone()
	end
	local first = m:narrow(1,1,index -1)
	local second = m:narrow(1, index + 1, nRows - index)
	return torch.cat(first, second, 1)
end

function getKeyset(map)
	local keyset={}
	local n=1
	for k,v in pairs(map) do
	  keyset[n]=k
	  n=n+1
	end
	return keyset
end

function maskRows(m, y)
	local nonZero = {}
	for i = 1,y:size(1) do
		print(y[i])
		if y[i] ~= 0 then
			table.insert(nonZero, i)
		end
	end
	return m:index(1,torch.LongTensor(nonZero))
end


function shuffle(a)
	local rnd,trem,getn,ins = math.random,table.remove,table.getn,table.insert;
    local r = {};
    while getn(a) > 0 do
        ins(r, trem(a, rnd(getn(a))));
    end
    return r;
end

function getParamLinScale(min, max)
	local x = math.random();
	return min + x * (max - min)
end

function getParamLogScale(min,  max)
	local x = math.random();
	return min * math.exp(math.log(max/min) * x);
end

function cloneOnce(net)
	local params, gradParams = net:parameters()
 	if params == nil then
		params = {}
	end
	local paramsNoGrad
	if net.parametersNoGrad then
		paramsNoGrad = net:parametersNoGrad()
	end
	local mem = torch.MemoryFile("w"):binary()
	mem:writeObject(net)
	-- We need to use a new reader for each clone.
	-- We don't want to use the pointers to already read objects.
	local reader = torch.MemoryFile(mem:storage(), "r"):binary()
	local clone = reader:readObject()
	reader:close()
	local cloneParams, cloneGradParams = clone:parameters()
	local cloneParamsNoGrad
	for i = 1, #params do
	  cloneParams[i]:set(params[i])
	  cloneGradParams[i]:set(gradParams[i])
	end
	if paramsNoGrad then
	  cloneParamsNoGrad = clone:parametersNoGrad()
	  for i =1,#paramsNoGrad do
	    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
	  end
	end
	mem:close()
	return clone
end

--[[ Creates clones of the given network.
The clones share all weights and gradWeights with the original network.
Accumulating of the gradients sums the gradients properly.
The clone also allows parameters for which gradients are never computed
to be shared. Such parameters must be returns by the parametersNoGrad
method, which can be null.
--]]
function cloneManyTimes(net, T)
print(collectgarbage("count"))
  local clones = {}
  local params, gradParams = net:parameters()
  if params == nil then
    params = {}
  end
  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
      cloneParamsNoGrad = clone:parametersNoGrad()
      for i =1,#paramsNoGrad do
        cloneParamsNoGrad[i]:set(paramsNoGrad[i])
      end
    end
    clones[t] = clone
    print('clone', t)
   -- print(collectgarbage("count"))
  end
  mem:close()
  collectgarbage()
  return clones
end


function run() 
	a = torch.randn(5,5)
	y = torch.zeros(5)
	y[2] = 1
	y[3] = 1
	m = maskRows(a, y)
	print(a)
	print(m)
end
