"""
paper: Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph
Reference From: https://github.com/pliang279/MFN & https://github.com/A2Zadeh/CMU-MultimodalSDK
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchsnooper
from itertools import chain, combinations

__all__ = ['Graph_MFN']

class DynamicFusionGraph(nn.Module):
	"""
	Input : (list): [InputModalA, InputModalB, ..., InputModalZ], singleton vertex representation.
	Output: (return a tuple in forward): (t_output, outputs, efficacies) 
            t_output   : i.e. T used as c_hat in MFN model.
            outputs    :   each node representation
            efficacies : the alpha list: [batch_size, 19] (19 for three modal fusion, 65 for four modal fusion)

	Examples:
	--------
	>>> from torch.autograd import Variable
	>>> import torch.nn.functional as F
	>>> import numpy
	>>> inputx = Variable(torch.Tensor(numpy.array(numpy.zeros([32, 40]))), requires_grad=True)
	>>> inputy = Variable(torch.Tensor(numpy.array(numpy.zeros([32, 12]))), requires_grad=True)
	>>> inputz = Variable(torch.Tensor(numpy.array(numpy.zeros([32, 20]))), requires_grad=True)
	>>> inputw = Variable(torch.Tensor(numpy.array(numpy.zeros([32, 25]))), requires_grad=True)
	>>> modalities = [inputx, inputy, inputz, inputw]
	>>> pattern_model = nn.Sequential(nn.Linear(100, 20))
	>>> efficacy_model = nn.Sequential(nn.Linear(100, 20))
	>>> fmodel = DynamicFusionGraph(pattern_model, [40, 12, 20, 25], 20, efficacy_model)
	>>> out = fmodel(modalities)
	"""
	def __init__(self, pattern_model, in_dimensions, out_dimension, efficacy_model, device):
		"""
        Args:
            pattern_model - nn.Module, a nn.Sequential model which will be used as core of the models inside the DFG
            in_dimensions - List, input dimensions of each modality
            out_dimension - int, output dimension of the pattern models
            efficacy_model - the core of the efficacy model
    	"""
		super(DynamicFusionGraph, self).__init__()
		self.num_modalities = len(in_dimensions)
		self.in_dimensions = in_dimensions
		self.out_dimension = out_dimension

        # in this part we sort out number of connections, how they will be connected etc.
        # powerset = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)] for three modal fusion. 
		self.powerset = list(
            chain.from_iterable(combinations(range(self.num_modalities), r) for r in range(self.num_modalities + 1)))[1:]
        
        # initializing the models inside the DFG
		self.input_shapes = {tuple([key]): value for key, value in zip(range(self.num_modalities), in_dimensions)}
        
		self.networks = {}
		self.total_input_efficacies = 0 # total_input_efficacies: for alpha list size = [batch_size, total_input_efficacies].
        
        # loop over n-modal node (n >= 2)
		for key in self.powerset[self.num_modalities:]:
            # connections coming from the unimodal components
			unimodal_dims = 0
			for modality in key:
				unimodal_dims += in_dimensions[modality]
			multimodal_dims = ((2 ** len(key) - 2) - len(key)) * out_dimension
			self.total_input_efficacies += 2 ** len(key) - 2
            # for the network that outputs key component, what is the input dimension
			final_dims = unimodal_dims + multimodal_dims
			self.input_shapes[key] = final_dims
			pattern_copy = copy.deepcopy(pattern_model)

            # final_model: transform the input to the node into out_dimension dim.
			final_model = nn.Sequential(
                *[nn.Linear(self.input_shapes[key], list(pattern_copy.children())[0].in_features), pattern_copy]).to(device)
			self.networks[key] = final_model
        # finished construction weights, now onto the t_network which summarizes the graph
		self.total_input_efficacies += 2 ** self.num_modalities - 1

		self.t_in_dimension = unimodal_dims + (2 ** self.num_modalities - (self.num_modalities) - 1) * out_dimension
		pattern_copy = copy.deepcopy(pattern_model)
		# self.t_network: generate top level representation τ 
		self.t_network = nn.Sequential(*[nn.Linear(self.t_in_dimension, list(pattern_copy.children())[0].in_features), pattern_copy]).to(device)
		# self.efficacy_model: generate the alpha list using the singleton vertex input. 
		# (in 3 modal [batch_size, dim_l+dim_v+dim_a] -> [batch_size, 19])
		self.efficacy_model = nn.Sequential(
            *[nn.Linear(sum(in_dimensions), list(efficacy_model.children())[0].in_features), efficacy_model,
              nn.Linear(list(efficacy_model.children())[-1].out_features, self.total_input_efficacies)]).to(device)

	def __call__(self, in_modalities):
		return self.fusion(in_modalities)

	def fusion(self, in_modalities):

		outputs = {}
		for modality, index in zip(in_modalities, range(len(in_modalities))):
			outputs[tuple([index])] = modality
		efficacies = self.efficacy_model(torch.cat([x for x in in_modalities], dim=1))
		efficacy_index = 0
		for key in self.powerset[self.num_modalities:]:
			small_power_set = list(chain.from_iterable(combinations(key, r) for r in range(len(key) + 1)))[1:-1]
			this_input = torch.cat([outputs[x] * efficacies[:, efficacy_index + y].view(-1, 1) for x, y in
									zip(small_power_set, range(len(small_power_set)))], dim=1)
			
			outputs[key] = self.networks[key](this_input)
			efficacy_index += len(small_power_set)

		small_power_set.append(tuple(range(self.num_modalities)))
		t_input = torch.cat([outputs[x] * efficacies[:, efficacy_index + y].view(-1, 1) for x, y in
								zip(small_power_set, range(len(small_power_set)))], dim=1)
		t_output = self.t_network(t_input)
		return t_output, outputs, efficacies

	def forward(self, x):
		print("Not yet implemented for nn.Sequential")
		exit(-1)


class Graph_MFN(nn.Module):
	def __init__(self, args):
		super(Graph_MFN, self).__init__()
		# print("Graph_MFN initialization ....")
		# print(args)
		self.d_l, self.d_a, self.d_v = args.feature_dims
		self.dh_l, self.dh_a, self.dh_v = args.hidden_dims
		total_h_dim = self.dh_l + self.dh_a + self.dh_v
		self.mem_dim = args.memsize
		self.inner_node_dim = args.inner_node_dim
		self.singleton_l_size, self.singleton_a_size, self.singleton_v_size = args.hidden_dims
		# Here Changed! (rm window_dim)
		# window_dim = args.windowsize
		output_dim = args.num_classes if args.train_mode == "classification" else 1
		# Here Changed! (rm attInShape, use inner_node_dim instead)
		# attInShape = total_h_dim * window_dim
		# gammaInShape = attInShape + self.mem_dim 
		gammaInShape = self.inner_node_dim + self.mem_dim  # Todo : we need get inner_node_dim from args.
		final_out = total_h_dim + self.mem_dim
		# h_att1 = args.NN1Config["shapes"]
		h_att2 = args.NNConfig["shapes"]
		h_gamma1 = args.gamma1Config["shapes"]
		h_gamma2 = args.gamma2Config["shapes"]
		h_out = args.outConfig["shapes"]
		# att1_dropout = args.NN1Config["drop"]
		att2_dropout = args.NNConfig["drop"]
		gamma1_dropout = args.gamma1Config["drop"]
		gamma2_dropout = args.gamma2Config["drop"]
		out_dropout = args.outConfig["drop"]

		self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
		self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
		self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

		# Here Changed! Todo : add Arg param singleton_l singleton_a singleton_v
		self.l_transform = nn.Linear(self.dh_l * 2, self.singleton_l_size)
		self.a_transform = nn.Linear(self.dh_a * 2, self.singleton_a_size)
		self.v_transform = nn.Linear(self.dh_v * 2, self.singleton_v_size)

		# Here Changed! (initialize the DFG part) Todo : add Arg param inner node dimension.
		pattern_model = nn.Sequential(nn.Linear(100, self.inner_node_dim)).to(args.device)
		efficacy_model = nn.Sequential(nn.Linear(100, self.inner_node_dim)).to(args.device) # Note : actually here inner_node_dim can change arbitrarily 
		self.graph_mfn = DynamicFusionGraph(pattern_model, [self.singleton_l_size, self.singleton_a_size, self.singleton_v_size], self.inner_node_dim, efficacy_model, args.device).to(args.device)
		# Here Changed!  (delete att1 )
		# self.att1_fc1 = nn.Linear(attInShape, h_att1)
		# self.att1_fc2 = nn.Linear(h_att1, attInShape)
		# self.att1_dropout = nn.Dropout(att1_dropout)

		# Here Changed! (alter the dim param.)
		self.att2_fc1 = nn.Linear(self.inner_node_dim, h_att2) # Note: might (inner_node_dim = self.mem_dim) is a common choice.
		self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
		self.att2_dropout = nn.Dropout(att2_dropout)

		self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
		self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
		self.gamma1_dropout = nn.Dropout(gamma1_dropout)

		self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
		self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
		self.gamma2_dropout = nn.Dropout(gamma2_dropout)

		self.out_fc1 = nn.Linear(final_out, h_out)
		self.out_fc2 = nn.Linear(h_out, output_dim)
		self.out_dropout = nn.Dropout(out_dropout)
		
	
	# @torchsnooper.snoop()
	def forward(self, text_x, audio_x, video_x):
		'''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)
            video_x: tensor of shape (batch_size, sequence_len, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
		'''
		text_x = text_x.permute(1,0,2)
		audio_x = audio_x.permute(1,0,2)
		video_x = video_x.permute(1,0,2)
		# x is t x n x d
		n = text_x.size()[1]
		t = text_x.size()[0]
		self.h_l = torch.zeros(n, self.dh_l).to(text_x.device)
		self.h_a = torch.zeros(n, self.dh_a).to(text_x.device)
		self.h_v = torch.zeros(n, self.dh_v).to(text_x.device)
		self.c_l = torch.zeros(n, self.dh_l).to(text_x.device)
		self.c_a = torch.zeros(n, self.dh_a).to(text_x.device)
		self.c_v = torch.zeros(n, self.dh_v).to(text_x.device)
		self.mem = torch.zeros(n, self.mem_dim).to(text_x.device)
		all_h_ls = []
		all_h_as = []
		all_h_vs = []
		# all_c_ls = []
		# all_c_as = []
		# all_c_vs = []
		all_mems = []
		all_efficacies = []
		for i in range(t):
			# prev time step (Here Changed !)
			# prev_c_l = self.c_l
			# prev_c_a = self.c_a
			# prev_c_v = self.c_v
			prev_h_l = self.h_l
			prev_h_a = self.h_a
			prev_h_v = self.h_v

			# curr time step
			new_h_l, new_c_l = self.lstm_l(text_x[i], (self.h_l, self.c_l))
			new_h_a, new_c_a = self.lstm_a(audio_x[i], (self.h_a, self.c_a))
			new_h_v, new_c_v = self.lstm_v(video_x[i], (self.h_v, self.c_v))
			# concatenate (Here Changed!)
			l_input = torch.cat([prev_h_l, new_h_l], dim=1)
			l_singleton_input = F.relu(self.l_transform(l_input))
			a_input = torch.cat([prev_h_a, new_h_a], dim=1)
			a_singleton_input = F.relu(self.a_transform(a_input))
			v_input = torch.cat([prev_h_v, new_h_v], dim=1)
			v_singleton_input = F.relu(self.v_transform(v_input))
			# Here Changed! (use DFG instead of attention.)
			# prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
			# new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
			# cStar = torch.cat([prev_cs,new_cs], dim=1)
			# attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
			# attended = attention*cStar

			# Note: we might want to record the efficacies for some reasons.
			attended, _, efficacies = self.graph_mfn([l_singleton_input, a_singleton_input, v_singleton_input])
			all_efficacies.append(efficacies.cpu().detach().squeeze().numpy())

			cHat = torch.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
			both = torch.cat([attended, self.mem], dim=1)
			gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
			gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
			self.mem = gamma1 * self.mem + gamma2 * cHat
			all_mems.append(self.mem)
			# update
			self.h_l, self.c_l = new_h_l, new_c_l
			self.h_a, self.c_a = new_h_a, new_c_a
			self.h_v, self.c_v = new_h_v, new_c_v
			all_h_ls.append(self.h_l)
			all_h_as.append(self.h_a)
			all_h_vs.append(self.h_v)
			# Here changed! all_c_ls is not used at all.
			# all_c_ls.append(self.c_l)
			# all_c_as.append(self.c_a)
			# all_c_vs.append(self.c_v)

		# last hidden layer last_hs is n x h
		last_h_l = all_h_ls[-1]
		last_h_a = all_h_as[-1]
		last_h_v = all_h_vs[-1]
		last_mem = all_mems[-1]
		last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
		output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
		res = {
			'M': output,
			'Efficacies': all_efficacies
		}
		return res
