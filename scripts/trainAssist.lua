require 'nn'
require 'optim'
require 'dataAssist'
require 'nngraph'
require 'RNNKhanCS'
require 'util'
require 'utilExp'
require 'lfs'

-- Singleton Vars
seconds_per_hour = 3600

start_epoch = 81

function run()
	local fileId = arg[1]
	assert(fileId ~= nil)

	math.randomseed(os.time())

	data = DataAssistMatrix()
	collectgarbage()

	local n_hidden = 200--math.floor(getParamLinScale(180, 220))
	local decay_rate = 1--0.99
	local init_rate = 30--getParamLinScale(0.008, 0.012)
	local mini_batch_size = 100--math.floor(getParamLinScale(50, 75))
	local dropoutPred = true
	local dropoutMem = false
	local max_grad = 5e-5 --math.random(15, 25)
	print('n_hidden' , n_hidden)
	print('init_rate', init_rate)
	print('decay_rate', decay_rate)
	print('mini_batch_size', mini_batch_size)
	print('dropoutPred', dropoutPred)
	print('dropoutMem', dropoutMem)
	print('maxGrad', max_grad)


	print('making rnn...')
	local rnn = RNNKhan{
		dropoutMem = dropoutMem,
		dropoutPred = dropoutPred,
		n_hidden = n_hidden,
		n_questions = data.n_questions,
		max_grad = max_grad,
		max_steps = 4290,
		--modelDir = '../output/trainRNNAssist/models/test_90'
	}
	--print(getAccuracy(rnn, data, mini_batch_size))


	lfs.mkdir('../output')
	lfs.mkdir('../output/trainRNNAssist')
	lfs.mkdir('../output/trainRNNAssist/models/')
	--print('acc', getAccuracy(rnn, data, mini_batch_size))
	print('rnn made!')

	
	local filePath = '../output/trainRNNAssist/' .. fileId .. '.txt'
	file = io.open(filePath, "w");
	file:write('n_hidden,' .. n_hidden .. '\n')
	file:write('init_rate,' .. init_rate .. '\n')
	file:write('decay_rate,' .. decay_rate .. '\n')
	file:write('mini_batch_size,' .. mini_batch_size .. '\n')
	file:write('dropoutPred,' .. tostring(dropoutPred) .. '\n')
	file:write('dropoutMem,' .. tostring(dropoutMem) .. '\n')
	file:write('maxGrad,' .. tostring(max_grad).. '\n')
	file:write('-----\n')
	file:write('i\taverageErr\ttestPred\trate\tclock\n')
	file:flush()
	trainMiniBatch(rnn, data, init_rate, decay_rate, mini_batch_size, file, fileId)
	file:close()
end

function trainMiniBatch(rnn, data, init_rate, decay_rate, mini_batch_size, file, modelId)
	print('train')
	local rate = init_rate
	local epochs = start_epoch
	local blob_size = 50
	--getAccuracy(rnn, data, mini_batch_size)
	while(true) do
		local startTime = os.time()
		local miniBatches = semiSortedMiniBatches(data:getTrainData(), blob_size, true)
		local totalTests = getTotalTests(miniBatches)
		collectgarbage()
		local sumErr = 0
		local numTests = 0
		local done = 0
		rnn:zeroGrad(350)
		local miniTests = 0
		local miniErr = 0
		for i,batch in ipairs(miniBatches) do
			local alpha = blob_size / totalTests
			--local alpha = blob_size / getNTests(batch)
			local err, tests, maxNorm = rnn:calcGrad(batch, rate, alpha)
			sumErr = sumErr + err
			numTests = numTests + tests
			collectgarbage()
			done = done + blob_size
			miniErr = miniErr + err
			miniTests = miniTests + tests
			if done % mini_batch_size == 0 then
				rnn:update(350, rate)
				rnn:zeroGrad(350)
				print('trainMini', i/#miniBatches, miniErr/miniTests, sumErr/numTests, maxNorm, rate)
				miniErr = 0
				miniTests = 0
				--rate = rate * decay_rate
			end
		end
		local avgErr = sumErr / numTests
		local testPred = getAccuracy(rnn, data, mini_batch_size);
		file:write(epochs .. '\t' .. avgErr .. '\t' .. testPred .. '\t' .. rate .. '\t' .. os.clock() .. '\n')
		file:flush()
		print(epochs, avgErr, testPred, rate, os.time() - startTime);
		rate = rate * decay_rate
		rnn:save('../output/trainRNNAssist/models/' .. modelId .. '_' .. epochs)
		epochs = epochs + 1
	end	
end

function getAuc(rnn, data)
end

function getAccuracy(rnn, data)
	local nTestBatchSize = 100
	local miniBatches = semiSortedMiniBatches(data:getTestData(), nTestBatchSize, false)
	local sumCorrect = 0
	local sumTested = 0
	for i,batch in ipairs(miniBatches) do
		local correct, tested, avgAcc = rnn:accuracyLight(batch)
		sumCorrect = sumCorrect + correct
		sumTested = sumTested + tested
		print('testMini', i/#miniBatches, sumCorrect/sumTested)
		collectgarbage()
	end
	return sumCorrect/sumTested
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

function checkExploding(grad)
	local norm = grad:norm()
	local max_grad = 10
	if norm > max_grad then
		alpha = max_grad / norm
		grad:mul(alpha)
    end
end

function updateMiniGrad(miniGrad, grad)
	if(miniGrad == nil) then 
		return grad;
	else 
		return miniGrad + grad;
	end
end

function baseline(trainData, testData)
	trainPctTrue = torch.sum(trainData) / trainData:nElement()
	testPctTrue = torch.sum(testData) / testData:nElement()
	if trainPctTrue > 0.5 then
		return testPctTrue
	else
		return 1 - testPctTrue 
	end
end


run()
