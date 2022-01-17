
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .MFN import MFN

__all__ = ['MFM']

def compute_kernel(x, y):
	x_size = x.size(0)
	y_size = y.size(0)
	dim = x.size(1)
	x = x.unsqueeze(1) # (x_size, 1, dim)
	y = y.unsqueeze(0) # (1, y_size, dim)
	tiled_x = x.expand(x_size, y_size, dim)
	tiled_y = y.expand(x_size, y_size, dim)
	kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
	return torch.exp(-kernel_input) # (x_size, y_size)

def loss_MMD(zy, args):
	zy_real_gauss = Variable(torch.randn(zy.size())) # no need to be the same size

	#if args.cuda:
	zy_real_gauss = zy_real_gauss.to(args.device)
	zy_real_kernel = compute_kernel(zy_real_gauss, zy_real_gauss)
	zy_fake_kernel = compute_kernel(zy, zy)
	zy_kernel = compute_kernel(zy_real_gauss, zy)
	zy_mmd = zy_real_kernel.mean() + zy_fake_kernel.mean() - 2.0*zy_kernel.mean()
	return zy_mmd

class encoderLSTM(nn.Module):
	def __init__(self, d, h): #, n_layers, bidirectional, dropout):
		super(encoderLSTM, self).__init__()
		self.lstm = nn.LSTMCell(d, h)
		self.fc1 = nn.Linear(h, h)
		self.h = h

	def forward(self, x, args):
		# x is t x n x h
		t = x.shape[0]
		n = x.shape[1]
		self.hx = torch.zeros(n, self.h).to(args.device)
		self.cx = torch.zeros(n, self.h).to(args.device)
		all_hs = []
		all_cs = []
		for i in range(t):
			self.hx, self.cx = self.lstm(x[i], (self.hx, self.cx))
			all_hs.append(self.hx)
			all_cs.append(self.cx)
		# last hidden layer last_hs is n x h
		last_hs = all_hs[-1]
		last_hs = self.fc1(last_hs)
		return last_hs

class decoderLSTM(nn.Module):
	def __init__(self, h, d):
		super(decoderLSTM, self).__init__()
		self.lstm = nn.LSTMCell(h, h)
		self.fc1 = nn.Linear(h, d)
		self.d = d
		self.h = h
		
	def forward(self, hT, t, args): # only embedding vector
		# x is n x d
		n = hT.shape[0]
		h = hT.shape[1]
		self.hx = torch.zeros(n, self.h).to(args.device)
		self.cx = torch.zeros(n, self.h).to(args.device)
		final_hs = []
		all_hs = []
		all_cs = []
		for i in range(t):
			if i == 0:
				self.hx, self.cx = self.lstm(hT, (self.hx, self.cx))
			else:
				self.hx, self.cx = self.lstm(all_hs[-1], (self.hx, self.cx))
			all_hs.append(self.hx)
			all_cs.append(self.cx)
			final_hs.append(self.hx.view(1,n,h))
		final_hs = torch.cat(final_hs, dim=0)
		all_recons = self.fc1(final_hs)
		return all_recons

class MFM(nn.Module):
	def __init__(self, args):
		super(MFM, self).__init__()
		self.d_l,self.d_a,self.d_v = args.feature_dims
		self.dh_l,self.dh_a,self.dh_v = args.hidden_dims
		self.args = args
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		zy_size = args.zy_size
		zl_size = args.zl_size
		za_size = args.za_size
		zv_size = args.zv_size
		fy_size = args.fy_size
		fl_size = args.fl_size
		fa_size = args.fa_size
		fv_size = args.fv_size
		zy_to_fy_dropout = args.zy_to_fy_dropout
		zl_to_fl_dropout = args.zl_to_fl_dropout
		za_to_fa_dropout = args.za_to_fa_dropout
		zv_to_fv_dropout = args.zv_to_fv_dropout
		fy_to_y_dropout = args.fy_to_y_dropout
		self.mem_dim = args.memsize
		window_dim = args.windowsize
		output_dim = args.num_classes if args.train_mode == "classification" else 1
		attInShape = total_h_dim*window_dim
		gammaInShape = attInShape+self.mem_dim
		final_out = total_h_dim+self.mem_dim
		h_att1 = args.NN1Config["shapes"]
		h_att2 = args.NN2Config["shapes"]
		h_gamma1 = args.gamma1Config["shapes"]
		h_gamma2 = args.gamma2Config["shapes"]
		h_out = args.outConfig["shapes"]
		att1_dropout = args.NN1Config["drop"]
		att2_dropout = args.NN2Config["drop"]
		gamma1_dropout = args.gamma1Config["drop"]
		gamma2_dropout = args.gamma2Config["drop"]
		out_dropout = args.outConfig["drop"]
		last_mfn_size = total_h_dim + args.memsize
		
		self.encoder_l = encoderLSTM(self.d_l,zl_size)
		self.encoder_a = encoderLSTM(self.d_a,za_size)
		self.encoder_v = encoderLSTM(self.d_v,zv_size)

		self.decoder_l = decoderLSTM(fy_size+fl_size,self.d_l)
		self.decoder_a = decoderLSTM(fy_size+fa_size,self.d_a)
		self.decoder_v = decoderLSTM(fy_size+fv_size,self.d_v)
		
		self.mfn_encoder = MFN(args)
		self.last_to_zy_fc1 = nn.Linear(last_mfn_size,zy_size)

		self.zy_to_fy_fc1 = nn.Linear(zy_size,fy_size)
		self.zy_to_fy_fc2 = nn.Linear(fy_size,fy_size)
		self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

		self.zl_to_fl_fc1 = nn.Linear(zl_size,fl_size)
		self.zl_to_fl_fc2 = nn.Linear(fl_size,fl_size)
		self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

		self.za_to_fa_fc1 = nn.Linear(za_size,fa_size)
		self.za_to_fa_fc2 = nn.Linear(fa_size,fa_size)
		self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

		self.zv_to_fv_fc1 = nn.Linear(zv_size,fv_size)
		self.zv_to_fv_fc2 = nn.Linear(fv_size,fv_size)
		self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

		self.fy_to_y_fc1 = nn.Linear(fy_size,fy_size)
		self.fy_to_y_fc2 = nn.Linear(fy_size,output_dim)
		self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)
		
	def forward(self, text_x, audio_x, video_x):
		'''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)
            video_x: tensor of shape (batch_size, sequence_len, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
		'''
		x_l = text_x.permute(1,0,2)
		x_a = audio_x.permute(1,0,2)
		x_v = video_x.permute(1,0,2)
		# x is t x n x d
		n = x_l.size()[1]
		t = x_l.size()[0]
		zl = self.encoder_l.forward(x_l, self.args)
		za = self.encoder_a.forward(x_a, self.args)
		zv = self.encoder_v.forward(x_v, self.args)

		mfn_last = self.mfn_encoder.forward(text_x, audio_x, video_x)['L']
		zy = self.last_to_zy_fc1(mfn_last)
		mmd_loss = loss_MMD(zl, self.args)+loss_MMD(za, self.args)+loss_MMD(zv, self.args)+loss_MMD(zy, self.args)
		missing_loss = 0.0

		fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
		fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
		fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
		fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))
		
		fyfl = torch.cat([fy,fl], dim=1)
		fyfa = torch.cat([fy,fa], dim=1)
		fyfv = torch.cat([fy,fv], dim=1)

		dec_len = t
		x_l_hat = self.decoder_l.forward(fyfl,dec_len, self.args)
		x_a_hat = self.decoder_a.forward(fyfa,dec_len, self.args)
		x_v_hat = self.decoder_v.forward(fyfv,dec_len, self.args)
		y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
		decoded = [x_l_hat,x_a_hat,x_v_hat,y_hat]

		return decoded,mmd_loss,missing_loss
