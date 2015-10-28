require 'nn'
require 'nngraph'
require 'util'
require 'utilExp'
require 'lfs'
local class = require 'class'

RNN = class.new('RNN')


function RNN:__init(params)
	self.n_questions = params['n_questions']
	self.n_hidden = params['n_hidden']
	self.use_dropout = params['dropout']
	self.max_grad = params['maxGrad']
	self.dropoutPred = params['dropoutPred']
	self.max_steps = params['maxSteps']

	self.n_input = self.n_questions * 2
	self.compressedSensing = params['compressedSensing']
	if(self.compressedSensing) then
		self.n_input = params['compressedDim']
		torch.manualSeed(12345)
		self.basis = torch.randn(self.n_questions * 2, self.n_input)
	end
	if(params['modelDir'] ~= nil) then
		self:load(params['modelDir'])
	else
		self:build(params)
	end
	print('rnn made')
end

function RNN:build(params)

	-- The transfer parameters
	local transfer = nn.Linear(self.n_hidden, self.n_hidden)

	-- The first layer 
	local start = nn.Linear(1, self.n_hidden)

	-- Prototypical layer
	local inputM = nn.Identity()(); -- the memory input
	local inputX = nn.Identity()(); -- the last student activity
	local inputY = nn.Identity()(); -- the next question answered
	local truth  = nn.Identity()(); -- whether the next question is correct

	local linM   = transfer:clone('weight', 'bias')(inputM);
	local linX   = nn.Linear(self.n_input, self.n_hidden)(inputX);
	local madd   = nn.CAddTable()({linM, linX});
	local hidden = nn.Tanh()(madd);
	
	local predInput = nil
	if(self.dropoutPred) then
		predInput = nn.Dropout()(hidden)
	else
		predInput = hidden
	end

	local linY = nn.Linear(self.n_hidden, self.n_questions)(predInput);
	local pred_output   = nn.Sigmoid()(linY);
	local pred          = nn.Sum(2)(nn.CMulTable()({pred_output, inputY}));
	local err           = nn.BCECriterion()({pred, truth})

	linX:annotate{name='linX'}
	linY:annotate{name='linY'}
	linM:annotate{name='linM'}

	local layer         = nn.gModule({inputM, inputX, inputY, truth}, {pred, err, hidden});

	self.start = start;
	self.layer = layer;

	self:rollOutNetwork()
end

function RNN:rollOutNetwork() 
	self.layers = self:cloneManyTimes(self.max_steps)
	for i = 1,self.max_steps do
		self.layers[i]:training()
	end
end

function RNN:zeroGrad(n_steps)
	self.start:zeroGradParameters()
	self.layer:zeroGradParameters()
end

function RNN:update(n_steps, rate)
	self.start:updateParameters(rate)
	self.layer:updateParameters(rate)
end

function RNN:fprop(batch)
	local n_steps = getNSteps(batch)
	local n_students = #batch
	assert(n_steps >= 1)
	assert(n_steps < self.max_steps)
	local inputs = {};
	local sumErr = 0;
	local numTests = 0
	local state = self.start:forward(torch.zeros(n_students, 1))
	for k = 1,n_steps do
		local inputX, inputY, truth = self:getInputs(batch, k)
		local mask = self:getMask(batch, k)
		inputs[k]  = {state, inputX, inputY, truth};
		local output = self.layers[k]:forward(inputs[k]);
		state = output[3]
		local stepErr = output[2][1] * n_students	-- scalar
		numTests = mask:sum() + numTests
		sumErr = sumErr + stepErr;
	end
	return sumErr, numTests, inputs
end	

function RNN:save(dir)
	lfs.mkdir(dir)
	torch.save(dir .. '/start.dat', self.start)
	torch.save(dir .. '/layer.dat', self.layer)
end

function RNN:load(dir)
	self.start = torch.load(dir .. '/start.dat')
	self.layer = torch.load(dir .. '/layer.dat')
	self:rollOutNetwork()
end

function RNN:calcGrad(batch, rate, alpha)
	local n_steps = getNSteps(batch)
	local n_students = #batch
	if(n_steps > self.max_steps) then
		print(n_steps, self.max_steps)
	end
	assert(n_steps <= self.max_steps)

	local maxNorm = 0
	local sumErr, numTests, inputs = self:fprop(batch)

	local parentGrad = torch.zeros(n_students, self.n_hidden)
	for k = n_steps,1,-1 do
		local layerGrad = self.layers[k]:backward(inputs[k], {
			torch.zeros(n_students),
			torch.ones(n_students):mul(alpha),
			parentGrad
		})
		parentGrad = layerGrad[1] -- because state is the first input
		
	end
	self.start:backward(torch.zeros(n_students, 1), parentGrad)
	return sumErr, numTests, maxNorm;
end

function RNN:getMask(batch, k)
	local mask = torch.zeros(#batch)
	for i,ans in ipairs(batch) do
		if(k + 1 <= ans['n_answers']) then
			mask[i] = 1
		end
	end
	return mask
end

function RNN:getInputs(batch, k)
	local n_students = #batch
	local mask = self:getMask(batch, k)
	local inputX = torch.zeros(n_students, 2 * self.n_questions)
	local inputY = torch.zeros(n_students, self.n_questions)
	local truth = torch.zeros(n_students)

	for i,answers in ipairs(batch) do
		if(k + 1 <= answers['n_answers']) then
			local currentId = answers['questionId'][k]
			local nextId    = answers['questionId'][k + 1] 
			local currentCorrect = answers['correct'][k]
			local nextCorrect    = answers['correct'][k + 1]
			
			local xIndex = self:getXIndex(currentCorrect, currentId)
			inputX[i][xIndex] = 1
			
			truth[i] = nextCorrect
			inputY[i][nextId] = 1
		end
	end
	--compressed sensing
	if(self.compressedSensing) then
		inputX = inputX * self.basis
	end
	return inputX, inputY, truth
end

function RNN:getXIndex(correct, id)
	assert(correct == 0 or correct == 1)
	assert(id ~= 0)
	local xIndex = correct * self.n_questions + id
	assert(xIndex >= 1)
	assert(xIndex ~= nil)
	assert(xIndex <= 2 * self.n_questions)
	return xIndex
end

function RNN:cloneManyTimes(n)
	return cloneManyTimes(self.layer, n)
end

function RNN:preventExplosion(grad)
	local norm = grad:norm()
	if norm > self.max_grad then
		print('explosion')
		local alpha = self.max_grad / norm
		grad:mul(alpha)
    end
    return norm
end

function RNN:err(batch)
	local n_steps = getNSteps(batch)
	for i = 1,n_steps do
		self.layers[i]:evaluate()
	end
	local sumErr, numTests, inputs = self:fprop(batch)
	for i = 1,n_steps do
		self.layers[i]:training()
	end
	return sumErr / numTests
end

function RNN:accuracy(batch)
	local n_steps = getNSteps(batch)
	local n_students = #batch

	self.layer:evaluate()

	local sumCorrect = 0
	local numTested = 0

	local state = self.start:forward(torch.zeros(n_students, 1))
	for k = 1,n_steps do
		local inputX, inputY, truth = self:getInputs(batch, k)
		inputX = inputX
		inputY = inputY
		local inputs = {state, inputX, inputY, truth};
		local output = self.layer:forward(inputs);
		state = output[3]:clone()
		
		local mask = self:getMask(batch, k)
		local p = output[1]:double()
		local pred = torch.gt(p, 0.5):double()
		local correct = torch.eq(pred, truth):double()
		local numCorrect = correct:cmul(mask):sum()
		sumCorrect = sumCorrect + numCorrect
		numTested = numTested + mask:sum()
	end

	-- for dropout
	for i = 1,n_steps do
		self.layer:training()
	end
	return sumCorrect, numTested
end

function RNN:getPredictionTruth(batch)
	local n_steps = getNSteps(batch)
	local n_students = #batch

	-- for dropout
	self.layer:evaluate()

	local predictionTruths = {}

	local state = self.start:forward(torch.zeros(n_students, 1))
	for k = 1,n_steps do
		local inputX, inputY, truth = self:getInputs(batch, k)
		inputX = inputX
		inputY = inputY
		local inputs = {state, inputX, inputY, truth};
		local output = self.layer:forward(inputs);
		state = output[3]:clone()
		
		local mask = self:getMask(batch, k)
		local pred = output[1]:double()
		
		for i = 1,n_students do
			if(mask[i] == 1) then
				predictionTruth = {
					pred = pred[i],
					truth = truth[i]
				}
				table.insert(predictionTruths, predictionTruth)
			end
		end
	end

	-- for dropout
	self.layer:training()
	return predictionTruths
end