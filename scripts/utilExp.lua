function getNTests(batch)
	local n_steps = getNSteps(batch)
	local n_students = #batch
	local m = torch.zeros(n_students, n_steps)
	for i = 1,n_steps do
		local mask = getMask(batch, i)
		for j = 1, n_students do
			m[j][i] = mask[j]
		end
	end
	return m:sum()
end

function getTotalTests(batches, maxSteps)
	local total = 0
	for i,batch in ipairs(batches) do
		local nTests = getNTests(batch)
		if(maxSteps ~= nil and maxSteps > 0) then
			nTests = math.min(maxSteps, nTests)
		end
		total = total + nTests
	end
	return total
end

function getNSteps(batch)
	local maxSteps = 0
	for i,ans in ipairs(batch) do
		if(ans['n_answers'] > maxSteps) then
			maxSteps = ans['n_answers']
		end
	end
	return maxSteps - 1
end

function getMask(batch, k)
	local mask = torch.zeros(#batch)
	for i,ans in ipairs(batch) do
		if(k + 1 <= ans['n_answers']) then
			mask[i] = 1
		end
	end
	return mask
end