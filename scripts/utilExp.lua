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

function evaluate(rnn, data)
	local miniBatches = semiSortedMiniBatches(data:getTestData(), 100, false)
	local allPredictions = {}
	local totalPositives = 0
	local totalNegatives = 0

	for i, batch in ipairs(miniBatches) do
		local batch = miniBatches[i]
		local pps = rnn:getPredictionTruth(batch)
		for i,prediction in ipairs(pps) do
			if(prediction['truth'] == 1) then
				totalPositives = totalPositives + 1
			else
				totalNegatives = totalNegatives + 1
			end

			table.insert(allPredictions, prediction)
		end
		collectgarbage()
	end

	function compare(a,b)
	  return a['pred'] > b['pred']
	end
	table.sort(allPredictions, compare)

	local truePositives = 0
	local falsePositives = 0
	local correct = 0
	local auc = 0
	local lastFpr = nil
	local lastTpr = nil
	for i,p in ipairs(allPredictions) do
		if(p['truth'] == 1) then
			truePositives = truePositives + 1
		else
			falsePositives = falsePositives + 1
		end

		local guess = 0
		if(p['pred'] > 0.5) then guess = 1 end
		if(guess == p['truth']) then correct = correct + 1 end

		local fpr = falsePositives / totalNegatives
		local tpr = truePositives / totalPositives
		if(i % 500 == 0) then
			if lastFpr ~= nil then
				local trapezoid = (tpr + lastTpr) * (fpr - lastFpr) *.5
				auc = auc + trapezoid
			end	
			lastFpr = fpr
			lastTpr = tpr
		end
		if(recall == 1) then break end
	end

	local accuracy = correct / #allPredictions
	return auc, accuracy
end

function semiSortedMiniBatches(dataset, mini_batch_size, trimToBatchSize)

	-- round down so that minibatches are the same size
	local trimmedAns = {}
	if(trimToBatchSize) then
		local nTemp = #dataset
		local maxNum = nTemp - (nTemp % mini_batch_size)
		local shuffled = shuffle(getKeyset(dataset))
		for i,s in ipairs(shuffled) do
			if(i <= maxNum) then
				table.insert(trimmedAns, dataset[s])
			end
		end
	else
		trimmedAns = dataset;
	end

	-- sort answers
	function compare(a,b)
	  return a['n_answers'] < b['n_answers']
	end
	table.sort(trimmedAns, compare)

	-- make minibatches
	local miniBatches = {}
	for j=1,#trimmedAns,mini_batch_size do
		miniBatch = {}
		for k = j, j + mini_batch_size - 1 do
			table.insert(miniBatch, trimmedAns[k])
		end
		table.insert(miniBatches, miniBatch)
	end

	-- shuffle minibatches
	local shuffledBatches = {}
	for i, s in ipairs(shuffle(getKeyset(miniBatches))) do
		table.insert(shuffledBatches, miniBatches[s])
	end

	return shuffledBatches
end