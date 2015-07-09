require 'csvigo'
local class = require 'class'

Data = class.new('Data')

-- Function: Get Data
-- Reads a CSV in standard feature format. Each row is a student
-- and each col is the score for the student on the
-- corresponding item.
function Data:__init(params) 
	local name = self:getName(params)
	print('Loading ' .. name)
	local path = '../data/synthetic/' .. name
	local rawData = csvigo.load({path=path, mode='raw', header=false})
	local totalStudents = table.getn(rawData)
	local n_questions = table.getn(rawData[1])
	local n_steps = n_questions - 1
	local n_students = totalStudents / 2
	local data = torch.Tensor(totalStudents, n_questions)
	for r,row in ipairs(rawData) do
		for c,v in ipairs(row) do 
			data[r][c] = v;        
		end
	end

	self.n_questions = n_questions
	self.n_students = n_students
	self.n_steps = n_steps

	self.nTest = self.n_students
	self.nTrain = self.n_students

	local trainData = torch.Tensor(data:narrow(1, 1, n_students))
	local testData = torch.Tensor(data:narrow(1, n_students, n_students))
	trainData = self:compressData(trainData)
	testData = self:compressData(testData)
	
	self.trainData = trainData
	self.testData = testData
end

function Data:getName(params)
	local name = 'naive';
	name = name .. '_c' .. params['c'];
	name = name .. '_q' .. params['q'];
	name = name .. '_s' .. params['n'];
	name = name .. '_v' .. params['v']
	name = name ..'.csv'
	return name
end

function Data:compressData(dataset)
	local newDataset = {}
	for i = 1,self.n_students do
		local answers = self:compressAnswers(dataset[i])
		table.insert(newDataset, answers)
	end
	return newDataset
end

function Data:compressAnswers(answers)
	local newAnswers = {}
	newAnswers['questionId'] = torch.zeros(self.n_questions)
	newAnswers['time'] = torch.zeros(self.n_questions)
	newAnswers['correct'] = torch.zeros(self.n_questions)
	newAnswers['n_answers'] = self.n_questions
	for i = 1, self.n_questions do
		newAnswers['questionId'][i] = i
		newAnswers['correct'][i] = answers[i]
	end
	return newAnswers
end

function Data:getTestData()
	return self.testData
end

function Data:getTrainData()
	return self.trainData
end