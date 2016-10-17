require 'nn'
require 'cunn'
require 'cudnn'

--nikolay's model + batch norm
function define_convolutional_model(input_size, number_of_filters)
  local model = nn.Sequential()
  model:add(nn.VolumetricConvolution(1, number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  model:add(nn.VolumetricBatchNormalization(number_of_filters))
  model:add(nn.ReLU())
  model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
  model:add(nn.VolumetricConvolution(number_of_filters, 2 * number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  model:add(nn.VolumetricBatchNormalization(2 * number_of_filters))
  model:add(nn.ReLU())
  model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
  model:add(nn.VolumetricConvolution(2 * number_of_filters, 4 * number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  model:add(nn.VolumetricBatchNormalization(4 * number_of_filters))
  model:add(nn.ReLU())
  model:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
  model:add(nn.Reshape(4 * number_of_filters * ((input_size / 8) ^ 3)))
  return model
end

function define_vgg_model(input_size, n_outputs, number_of_filters, number_of_scales, number_of_rotations)
  local convolutional_model = define_convolutional_model(input_size, number_of_filters) 
  local scale_parallel_model = nn.Parallel(2, 2)
  for scale_index = 1, number_of_scales do
    scale_parallel_model:add(convolutional_model:clone())
  end
  local model_for_single_rotation = nn.Sequential()
  model_for_single_rotation:add(scale_parallel_model)
  model_for_single_rotation:add(nn.Reshape(1, 4 * number_of_filters * number_of_scales * ((input_size / 8) ^ 3)))
  local rotation_parallel_model = nn.Parallel(2, 2)
  for rotation_index = 1, number_of_rotations do
    rotation_parallel_model:add(model_for_single_rotation:clone())
  end
  local full_model = nn.Sequential()
  full_model:add(rotation_parallel_model)
  full_model:add(nn.SpatialMaxPooling(1, number_of_rotations))
  full_model:add(nn.Reshape(4 * number_of_filters * number_of_scales * ((input_size / 8) ^ 3)))
  local kFullyConnectedMultiplier = 128
  full_model:add(nn.Linear(4 * number_of_filters * number_of_scales * ((input_size / 8) ^ 3), kFullyConnectedMultiplier * number_of_filters))
  full_model:add(nn.ReLU())
  full_model:add(nn.Dropout())
  full_model:add(nn.Linear(kFullyConnectedMultiplier * number_of_filters, n_outputs))
  full_model:add(nn.LogSoftMax())
  full_model = full_model:cuda()
  full_model = cudnn.convert(full_model, cudnn)
  -- sharing parameters
  for rotation_index = 2, number_of_rotations do
    local current_module = full_model:get(1):get(rotation_index)
    current_module:share(full_model:get(1):get(1), 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  full_model:training()
  print(full_model)
  return full_model
end

--voxception block from brock et al.
function voxception_block(input_planes, number_of_filters)
  -- convolution 3x3x3
  local model1 = nn.Sequential()
  model1:add(nn.VolumetricConvolution(input_planes, input_planes * number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  --model1:add(nn.VolumetricBatchNormalization(input_planes * number_of_filters))
  -- convolution 1x1x1
  local model2 = nn.Sequential()
  model2:add(nn.VolumetricConvolution(input_planes, input_planes * number_of_filters, 1, 1, 1, 1, 1, 1, 0, 0, 0))
  --model2:add(nn.VolumetricBatchNormalization(input_planes * number_of_filters))
  -- concatinate models (same input, concatenate outputs)
  local model_conc = nn.Concat(input_planes * number_of_filters)
  model_conc:add(model1)
  model_conc:add(model2)
  local model_join = nn.Sequential()
  model_join:add(model_conc)
  model_join:add(nn.JoinTable(1))
  return model_join
  --return model1
end

--voxception downsampling block from brock et al.
function voxception_downsample_block(input_planes, number_of_filters)
  -- convolution 3x3x3, max pooling
  local model1 = nn.Sequential()
  model1:add(nn.VolumetricConvolution(input_planes, input_planes * number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))
  model1:add(nn.VolumetricBatchNormalization(input_planes * number_of_filters))
  model1:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
  -- convolution 3x3x3, av. pooling
  local model2 = nn.Sequential()
  model2:add(nn.VolumetricConvolution(input_planes, input_planes * number_of_filters, 3, 3, 3, 1, 1, 1, 1, 1, 1))

  model2:add(nn.VolumetricBatchNormalization(input_planes * number_of_filters))
  model2:add(nn.VolumetricAveragePooling(2,2,2,2,2,2))
  -- strided convolution 3x3x3, stride 2 
  local model3 = nn.Sequential()
  model3:add(nn.VolumetricConvolution(input_planes, input_planes * number_of_filters, 3, 3, 3, 2, 2, 2, 1, 1, 1))
  model3:add(nn.VolumetricBatchNormalization(input_planes * number_of_filters))
  -- strided convolution 1x1x1, stride 2 
  local model4 = nn.Sequential()
  model4:add(nn.VolumetricConvolution(input_planes, input_planes * number_of_filters, 1, 1, 1, 2, 2, 2, 0, 0, 0))
  model4:add(nn.VolumetricBatchNormalization(input_planes * number_of_filters))
  -- parallel models (same input, concatenate outputs)
  local model = nn.Concat(input_planes * number_of_filters)
  model:add(model1)
  model:add(model2)
  model:add(model3)
  model:add(model4)
  return model 
end

--voxception resnet block from brock et al.
function voxception_resnet_block(input_planes, number_of_filters, stride)
  local n_output1 = input_planes * number_of_filters / 4;
  local n_output2 = input_planes * number_of_filters / 2;
  --3x3x3 convolution, 3x3x3 convolution
  local m1 = nn.Sequential()
  m1:add(nn.VolumetricConvolution(input_planes,(n_output1),3,3,3,1,1,1,1,1,1))
  m1:add(nn.VolumetricBatchNormalization((n_output1)))
  m1:add(nn.VolumetricConvolution(n_output1,n_output2,3,3,3, stride,stride,stride, 1,1,1))
  m1:add(nn.VolumetricBatchNormalization((n_output2)))

  --1x1x1 convolution, 3x3x3 convolution, 1x1x1 convolution
  local m2 = nn.Sequential()
  m2:add(nn.VolumetricConvolution(input_planes, (n_output1),1,1,1,1,1,1,0,0,0))
  m2:add(nn.VolumetricBatchNormalization((n_output1)))
  m2:add(nn.VolumetricConvolution((n_output1), (n_output1),3,3,3, stride,stride,stride, 1,1,1))
  m2:add(nn.VolumetricBatchNormalization((n_output1)))
  m2:add(nn.VolumetricConvolution((n_output1), (n_output2),1,1,1,1,1,1,0,0,0))
  m2:add(nn.VolumetricBatchNormalization((n_output2)))

  --concatinate different models
  local s = nn.Concat(n_output2)
  s:add(m1)
  s:add(m2)

  --resnet (compute residuals)
  return nn.Sequential()
    :add(nn.ConcatTable()
      :add(s)
      :add(nn.VolumetricConvolution(input_planes, n_output2, 1,1,1, stride,stride,stride, 0,0,0))) --TODO: use nn.Identity()?
    :add(nn.CAddTable())
end

--voxception model from brock et al.
function define_voxception_model(input_size, n_outputs, model_creation_input, initial_nr_planes, number_of_filters, number_of_blocks, number_of_scales, number_of_rotations)
  local is = model_creation_input:size()
  print(is)
  -- voxception model: single scale, single rotation
  local voxception_model = nn.Sequential()
  local number_planes = 1
  local prod_strides = 1
  --input layer (first block)
  local conv_block = voxception_block(number_planes, initial_nr_planes)
  number_planes = initial_nr_planes * 2
  voxception_model:add(conv_block)
  for block_index = 2, number_of_blocks do
    --downsample block
    downsample_block = voxception_downsample_block(number_planes, 1)
    voxception_model:add(downsample_block)

    --conv block
    conv_block = voxception_block(number_planes, number_of_filters)
    number_planes = number_planes * number_of_filters * 2
    prod_strides = prod_strides
    voxception_model:add(conv_block)
  end
  -- multi scale, single rotation
  local scale_parallel_model = nn.Parallel(2, 2)
  for scale_index = 1, number_of_scales do
    scale_parallel_model:add(voxception_model:clone())
  end

  -- multi scale, multiple rotations
  local twidth = number_planes * number_of_scales * ((input_size/prod_strides)^3)
  local model_for_single_rotation = nn.Sequential()
  model_for_single_rotation:add(scale_parallel_model)
  model_for_single_rotation:add(nn.Reshape(1, twidth))
  local rotation_parallel_model = nn.Parallel(2, 2)
  for rotation_index = 1, number_of_rotations do
    rotation_parallel_model:add(model_for_single_rotation:clone())
  end
  --

  -- fully connected layers on top
  local full_model = nn.Sequential()
  full_model:add(rotation_parallel_model)
  full_model:add(nn.SpatialMaxPooling(1, number_of_rotations))
  full_model:add(nn.JoinTable(1))
  local kFullyConnectedMultiplier = 128
  full_model:add(nn.Linear(twidth, kFullyConnectedMultiplier * number_planes))
  full_model:add(nn.ReLU())
  full_model:add(nn.Dropout())
  full_model:add(nn.Linear(kFullyConnectedMultiplier * number_planes, n_outputs))
  full_model:add(nn.LogSoftMax())
  full_model = full_model:cuda()
  --full_model = cudnn.convert(full_model, cudnn)
  ----[[ sharing parameters
  for rotation_index = 2, number_of_rotations do
    local current_module = full_model:get(1):get(rotation_index)
    current_module:share(full_model:get(1):get(1), 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  --]]
  full_model:training()
  print(full_model)
  return full_model
end
