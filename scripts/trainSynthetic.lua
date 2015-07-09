require 'dataSynthetic'
require 'rnn'
require 'lfs'

seconds_per_hour = 3600

outputRoot = "../output/synthetic/"
CONCEPT_NUM = 5
VERSION = 1

function run() 
	math.randomseed(os.time())

	local n_hidden = 400
	local decay_rate = 1.0
	local init_rate = 0.5
	local mini_batch_size = 50
	local blob_size = mini_batch_size

	local data = Data{n=4000,q=50,c=CONCEPT_NUM,v=VERSION}
	
	local rnn = RNN{
		dropout = true,
		n_hidden = n_hidden,
		n_questions = data.n_questions,
		max_grad = 100,
		max_steps = data.n_questions,
		--modelDir = outputRoot .. '/models/result_c5_v0_98'
	}

	local name = "result_c" .. CONCEPT_NUM .. "_v" .. VERSION
	lfs.mkdir(outputRoot)
	lfs.mkdir(outputRoot .. "models")

    file = io.open(outputRoot .. name .. ".txt", "w");
    print(dropout)
    file:write('version,', VERSION, '\n')
    file:write('n_hidden,', n_hidden, '\n');
    file:write('init_rate,' , init_rate, '\n');
    file:write('decay_rate,', decay_rate, '\n');
    file:write('dropout,', tostring(dropout), '\n');
    file:write('mini_batch_size,', mini_batch_size, '\n');
    trainMiniBatch(rnn, data, init_rate, decay_rate, mini_batch_size, blob_size, file, name)
    file:close()
	
end

function trainMiniBatch(rnn, data, init_rate, decay_rate, mini_batch_size, blob_size, file, modelId)
	print('train')
	local rate = init_rate
	local epochs = 1
	--getAccuracy(rnn, data, mini_batch_size)
	while(true) do
		local startTime = os.time()
		local miniBatches = semiSortedMiniBatches(data:getTrainData(), blob_size, true)
		local totalTests = getTotalTests(miniBatches)
		collectgarbage()
		local sumErr = 0
		local numTests = 0
		local done = 0
		rnn:zeroGrad()
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
				rnn:update(rate)
				rnn:zeroGrad()
				print(i/#miniBatches, sumErr/numTests)
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
		rnn:save(outputRoot .. 'models/' .. modelId .. '_' .. epochs)
		epochs = epochs + 1
	end	
end

function getAccuracy(rnn, data)
	local nTest = math.floor(data.nTest / 50)
	local miniBatches = semiSortedMiniBatches(data:getTestData(), nTest, false)
	local sumCorrect = 0
	local sumTested = 0
	for i,batch in ipairs(miniBatches) do
		local correct, tested, avgAcc = rnn:accuracy(batch)
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
run()