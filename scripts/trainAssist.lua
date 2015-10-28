require 'nn'
require 'optim'
require 'dataAssist'
require 'nngraph'
require 'rnn'
require 'util'
require 'utilExp'
require 'lfs'

START_EPOCH = 1
OUTPUT_DIR = '../output/trainRNNAssist/'
LEARNING_RATES = {30, 30, 30, 10, 10, 10, 5, 5, 5}
LEARNING_RATE_REPEATS = 4
MIN_LEARNING_RATE = 1

function run()
	local fileId = arg[1]
	assert(fileId ~= nil)

	math.randomseed(os.time())

	data = DataAssistMatrix()
	collectgarbage()

	local n_hidden = 200
	local decay_rate = 1 
	local init_rate = 30
	local mini_batch_size = 100
	local dropoutPred = true
	local max_grad = 5e-5 

	print('n_hidden' , n_hidden)
	print('init_rate', init_rate)
	print('decay_rate', decay_rate)
	print('mini_batch_size', mini_batch_size)
	print('dropoutPred', dropoutPred)
	print('maxGrad', max_grad)


	print('making rnn...')
	local rnn = RNN{
		dropoutPred = dropoutPred,
		n_hidden = n_hidden,
		n_questions = data.n_questions,
		maxGrad = max_grad,
		maxSteps = 4290,
		compressedSensing = true,
		compressedDim = 100
	}
	print('rnn made!')

	lfs.mkdir('../output')
	lfs.mkdir(OUTPUT_DIR)
	lfs.mkdir(OUTPUT_DIR .. 'models/')
	local filePath = OUTPUT_DIR  .. fileId .. '.txt'
	file = io.open(filePath, "w");
	file:write('n_hidden,' .. n_hidden .. '\n')
	file:write('init_rate,' .. init_rate .. '\n')
	file:write('decay_rate,' .. decay_rate .. '\n')
	file:write('mini_batch_size,' .. mini_batch_size .. '\n')
	file:write('dropoutPred,' .. tostring(dropoutPred) .. '\n')
	file:write('maxGrad,' .. tostring(max_grad).. '\n')
	file:write('-----\n')
	file:write('i\taverageErr\tauc\ttestPred\trate\tclock\n')
	file:flush()
	trainMiniBatch(rnn, data, mini_batch_size, file, fileId)
	file:close()
end

function trainMiniBatch(rnn, data, mini_batch_size, file, modelId)
	print('train')
	local epochIndex = START_EPOCH
	local blob_size = 50
	--getAccuracy(rnn, data, mini_batch_size)
	while(true) do
		local rate = getLearningRate(epochIndex)
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
				print('trainMini', i/#miniBatches, miniErr/miniTests, sumErr/numTests, rate)
				miniErr = 0
				miniTests = 0
				--rate = rate * decay_rate
			end
		end
		local auc, accuracy = evaluate(rnn, data)
		local avgErr = sumErr / numTests
		local outline = epochIndex .. '\t' 
		outline = outline .. avgErr .. '\t' 
		outline = outline .. auc .. '\t'
		outline = outline .. accuracy .. '\t' 
		outline = outline .. rate .. '\t' 
		outline = outline .. os.clock()
		file:write(outline .. '\n')
		file:flush()
		print(outline);
		rnn:save(OUTPUT_DIR .. 'models/' .. modelId .. '_' .. epochIndex)
		epochIndex = epochIndex + 1
	end	
end

function getLearningRate(epochIndex)
	local rate = MIN_LEARNING_RATE
	local rateIndex = math.floor((epochIndex - 1) / LEARNING_RATE_REPEATS) + 1
	if(rateIndex <= #LEARNING_RATES) then
		rate = LEARNING_RATES[rateIndex]
	end
	return rate
end


run()
