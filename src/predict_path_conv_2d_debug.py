
from operator import le
from numpy.lib.function_base import average
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib
import json
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import time
import math


map_size = str(8)

# input_path = '../data_2/maze_10x10_rnd/'
# input_path = '../data_2/maze_15x15_rnd/'
# input_path = '../data_2/maze_20x20_rnd/'
# input_path = '../data_2/maze_30x30_rnd/'
input_path = "../resources/dat_files/test_maps/" + map_size + "_single/"

# model_file = '../results_2/model_10_conv.hf5'
# model_file = '../results_2/model_15_conv.hf5'
# model_file = '../results_2/model_20_conv.hf5'
# model_file = '../results_2/model_30_conv.hf5'
model_file = '../results_2/model_2d_30k_combined_2.hf5'

# model_file = './model_2d_urf.hf5'

# ind_test = np.array(range(28000,29998)) #the range of samples for testing (the last 2000 samples are used)



ind_test = np.array(range(0,1499))

def pathlength(x,y):
    n = len(x) 
    lv = [math.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) for i in range (1,n)]
    L = sum(lv)
    return L

def open_astar_paths(path):
    '''
    Used to open a single json for astar paths
    '''
    with open(str(path)) as json_file:
        data = json.load(json_file)
        return data

def deviation(mp_nr,pred_path_x,pred_path_y,astar_length,astar_trace_x,astar_trace_y):
	
	# print("x: ", pred_path_x)
	# print("y: ", pred_path_y)


	
	cnpp_length = pathlength(pred_path_x,pred_path_y)
	astar_length_2 = pathlength(astar_trace_x,astar_trace_y)


	print("cnpp: ", cnpp_length)
	print("astar: ", astar_length)
	print("astar local: ", astar_length_2)

	print("Astar x y", astar_trace_x[0],astar_trace_y[0])
	print("Pred path x y", pred_path_x[0],pred_path_y[0])


	if astar_length == 0 or cnpp_length == 0:
		pass
	else:
		dev = -((astar_length-cnpp_length)/astar_length)* 100
		print(mp_nr, "Deviation: ", dev)

		return dev




#######################################################################################################################
def reconstruct_path(y_pred,env,s,g):

	sx=s[0]+1
	sy=s[1]+1
	gx=g[0]+1
	gy=g[1]+1
	#find predicted path
	y_pred[env==1]=-1
	pred_map=-1*np.ones((n+2,n+2))
	pred_map[1:n+1,1:n+1]=y_pred
	pred_map_2=-1*np.ones((n+2,n+2))
	pred_map_2[1:n+1,1:n+1]=y_pred

	pred_map[sx,sy]=0
	pred_map_2[gx,gy]=0

	T=pow(n,2)

	px_pred=np.zeros(T,dtype='int')
	py_pred=np.zeros(T,dtype='int')
	px_pred[0]=sx
	py_pred[0]=sy

	px_pred_2=np.zeros(T,dtype='int')
	py_pred_2=np.zeros(T,dtype='int')
	px_pred_2[0]=gx
	py_pred_2[0]=gy

	path_found=False
	got_stuck=False
	got_stuck_2=False

	for k in range(T):
		if not got_stuck:
			if math.sqrt(pow(px_pred[k]-gx,2)+pow(py_pred[k]-gy,2)) <= math.sqrt(2):
				px_pred[k+1]=gx
				py_pred[k+1]=gy
				px_pred=px_pred[0:k+2]
				py_pred=py_pred[0:k+2]
				path_found=True
				# print('forward path')
				break
			
			tmp=pred_map[px_pred[k]-1:px_pred[k]+2,py_pred[k]-1:py_pred[k]+2]		
			ii, jj = np.unravel_index(tmp.argmax(), tmp.shape)
			px_pred[k+1]=px_pred[k]+ii-1
			py_pred[k+1]=py_pred[k]+jj-1
			pred_map[px_pred[k+1],py_pred[k+1]]=0          
			
			h=np.array(np.where((px_pred_2==px_pred[k+1])&(py_pred_2==py_pred[k+1]))).squeeze()
			if h.size==1:
				px_pred_2=px_pred_2[0:h]
				py_pred_2=py_pred_2[0:h]
				px_pred=px_pred[0:k+2]
				py_pred=py_pred[0:k+2]
				px_pred=np.append(px_pred,np.flip(px_pred_2))
				py_pred=np.append(py_pred,np.flip(py_pred_2))
				path_found=True
				# print('forward cross')
				break
			
			if (px_pred[k+1]==px_pred[k]) and (py_pred[k+1]==py_pred[k]):
				px_pred=px_pred[0:px_pred.size-1]
				py_pred=py_pred[0:py_pred.size-1]
				got_stuck=True

		if not got_stuck_2:
			if math.sqrt(pow(px_pred_2[k]-sx,2)+pow(py_pred_2[k]-sy,2)) <= math.sqrt(2):
				px_pred_2[k+1]=sx
				py_pred_2[k+1]=sy
				px_pred_2=px_pred_2[0:k+2]
				py_pred_2=py_pred_2[0:k+2]
				px_pred = np.flip(px_pred_2)
				py_pred = np.flip(py_pred_2)
				path_found=True
				# print('backward path')
				break
			
			tmp=pred_map_2[px_pred_2[k]-1:px_pred_2[k]+2,py_pred_2[k]-1:py_pred_2[k]+2]
			ii, jj = np.unravel_index(tmp.argmax(), tmp.shape)
			px_pred_2[k+1]=px_pred_2[k]+ii-1
			py_pred_2[k+1]=py_pred_2[k]+jj-1
			pred_map_2[px_pred_2[k+1],py_pred_2[k+1]]=0
			
			h=np.array(np.where((px_pred==px_pred_2[k+1])&(py_pred==py_pred_2[k+1]))).squeeze()
			if h.size==1:
				px_pred=px_pred[0:h]
				py_pred=py_pred[0:h]
				px_pred_2=px_pred_2[0:k+2]
				py_pred_2=py_pred_2[0:k+2]
				px_pred=np.append(px_pred,np.flip(px_pred_2))
				py_pred=np.append(py_pred,np.flip(py_pred_2))
				path_found=True
				# print('backward cross')
				break
			
			if (px_pred_2[k+1]==px_pred_2[k]) and (py_pred_2[k+1]==py_pred_2[k]):
				px_pred_2=px_pred_2[0:px_pred_2.size-1]
				py_pred_2=py_pred_2[0:py_pred_2.size-1]
				got_stuck_2=True      

		if got_stuck and got_stuck_2:
			break

	###remove loops
	ind=np.array([])
	for k in range(px_pred.size):
		h=np.array( np.where( (px_pred==px_pred[k])&(py_pred==py_pred[k]) ) ).squeeze()
		if h.size>=2:
			ind=np.append(ind, np.array(range(h[0]+1,h[1]+1)))
	# print("5555", ind)
	
	px_pred=np.delete(px_pred,ind.astype(int))
	py_pred=np.delete(py_pred,ind.astype(int))
	# print("hello")
	###remove zig-zags
	while 1:
		k=0
		ind=np.array([])
		for i in range(px_pred.size-2):
			if math.sqrt(pow(px_pred[k]-px_pred[k+2],2)+pow(py_pred[k]-py_pred[k+2],2)) <= math.sqrt(2):
				ind=np.append(ind, k+1)
				k=k+2
			else:
				k=k+1

			if k>=px_pred.size-2:
				break
		
		if ind.size:
			# print("55555", ind)
			px_pred=np.delete(px_pred,ind.astype(int))
			py_pred=np.delete(py_pred,ind.astype(int))
		else:
			break
	# print("xxxxxxx", px_pred, py_pred)
	return px_pred, py_pred, path_found
#######################################################################################################################


print('Load data ...')
x = np.loadtxt(input_path + 'inputs.dat')
s_map = np.loadtxt(input_path + 's_maps.dat')
g_map = np.loadtxt(input_path + 'g_maps.dat')
y = np.loadtxt(input_path + 'outputs.dat')

print('Transform data ...')
x=x[ind_test,:]
s_map=s_map[ind_test,:]
g_map=g_map[ind_test,:]
y=y[ind_test,:]

m = x.shape[0]
n = int(np.sqrt(x.shape[1]))

x = x.reshape(m,n,n)
s_map = s_map.reshape(m,n,n)
g_map = g_map.reshape(m,n,n)

x_test=np.zeros((m,n,n,3))
x_test[:,:,:,0]=x
x_test[:,:,:,1]=s_map
x_test[:,:,:,2]=g_map
del x, s_map, g_map

y_test = y.reshape(m,n,n)
del y

print('Load model ...')
model=load_model(model_file)
model.summary()
#do prediction ones to initialize GPU
model.predict(np.zeros((1,n,n,3)))

metrics_time = []
metrics_success = []
metrics_deviation = []


file_path_lengths_urf = "/home/hussein/Desktop/pb-cnpp/pb-cnpp/resources/jsons/test_maps/" + map_size + "/lengths_urf.json"

file_path_trace_urf = "/home/hussein/Desktop/pb-cnpp/pb-cnpp/resources/jsons/test_maps/" + map_size + "/paths_urf.json"

with open(str(file_path_lengths_urf)) as json_file:
	length_data_urf = json.load(json_file)

astar_paths_2 = open_astar_paths(file_path_trace_urf)

urf_flag = True
u_flag = 0
h_flag = 0
print('Predict path ...')
print("M = ", m)
for i in range(m):
	# print(i)
	print("\n New Map: ", i,  "_________________________")
	astar_trace_x = []
	astar_trace_y = []
	### Predict path
	a = time.process_time()
	y_pred=model.predict(x_test[i,:,:,:].reshape(1,n,n,3)).squeeze()
	pred_t = time.process_time() - a
	# print('Prediction time: ', pred_t)
	
	### Reconstruct path from the prediction
	env=x_test[i,:,:,0].squeeze()
	s=np.array(np.nonzero(x_test[i,:,:,1].squeeze()==1))
	g=np.array(np.nonzero(x_test[i,:,:,2].squeeze()==1))
	
	a = time.process_time()
	path_found_flag = False
	# print("_________________________", y_pred)
	px,py,path_found_flag=reconstruct_path(y_pred,env,s,g)
	print("Num: ", i)
	print("Path Found:", path_found_flag)
	print("Px_start: ", px[0])
	print("Py_start: ", py[0])



	rec_t = time.process_time() - a
	# print('Reconstruction time: ', rec_t)
	metrics_time.append(pred_t+rec_t)	



	map_length = length_data_urf[i]
		# print("Map = URF")






	if path_found_flag:
		metrics_success.append(1)
		dev = deviation(i,px,py,map_length,astar_trace_x,astar_trace_y)
		if dev:
			metrics_deviation.append(dev)
	else:
		metrics_success.append(0)


	### Plot path 
	t=np.array(np.nonzero(y_test[i,:,:].squeeze()==1)) #ground-truth path
	t_pred=np.array(np.nonzero(y_pred>0.001)) #predicted path
	
	fig = plt.matshow(env.T,cmap='binary')
	plt.plot(t[0],t[1],'kx',markersize=15)
	for j in range(t_pred.shape[1]):
		plt.plot(t_pred[0,j],t_pred[1,j],'b.',markersize=15*y_pred[t_pred[0,j],t_pred[1,j]]+1)
	plt.plot(px-1,py-1,'b')
	plt.plot(s[0],s[1],'g.',markersize=15)
	plt.plot(g[0],g[1],'r.',markersize=15)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	# plt.show()
	# plt.savefig('../results_2/images/env%05d.png' % (i+1))	
	plt.close()




print("\n Success Rate:", (sum(metrics_success)/len(metrics_success))*100)
print("\n Avg Time:", average(metrics_time))
# print("\n", metrics_deviation)
# print("length of dev:", len(metrics_deviation))
# print("sum: ", sum(metrics_deviation))
print("\n Avg Deviation: ", sum(metrics_deviation)/len(metrics_deviation))

	




