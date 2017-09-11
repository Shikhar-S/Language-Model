from __future__ import generators
import tensorflow as tf
import time
import numpy as np


f='tinyshakespeare.txt'
vocab=set()
vocab_size=0
def format_file(f):
	write_file=open('format_allines.txt','w')
	with open(f,'rw') as F:
		for line in F:
			line=line.lower()
			for x in line:
				if(x not in vocab):
					vocab.add(x)
			write_file.write(line[1:-2])
	write_file.close()

format_file(f)
vocab_size=len(vocab)
idx_to_vocab=dict(enumerate(vocab))
vocab_to_idx=dict(zip(idx_to_vocab.values(),idx_to_vocab.keys()))

with open('format_allines.txt','r') as F:
	raw_data=F.read()
	data=[vocab_to_idx[c.lower()] for c in raw_data]

print vocab_size
print len(data)
del raw_data


def reset_graph():
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()

def gen_epochs(num_steps,batch_size,epochs=-1):
	start=0
	end=0
	ret=list()
	if(len(data)< (epochs*batch_size)):
		print 'Exceeding limits'
		epochs=(len(data))//(batch_size)
	if epochs==-1:
		epochs=(len(data)-num_steps)//(batch_size)
		if(epochs>10000):
			epochs=10000
		print epochs
	for i in range(epochs):
		ret_x=list()
		ret_y=list()
		for batch in xrange(batch_size):
			end=start+num_steps
			ret_x.append(data[start:end])
			ret_y.append(data[start+1:end+1])
			start=start+1
		# ret_tens_x=tf.convert_to_tensor_or_sparse_tensor(ret_x,dtype=tf.int32)
		# ret_tens_y=tf.convert_to_tensor_or_sparse_tensor(ret_y,dtype=tf.int32)
		ret_tens_x=np.asarray(ret_x)
		ret_tens_y=np.asarray(ret_y)
		yield (ret_tens_x,ret_tens_y)

def build_graph(state_size=100,num_classes=vocab_size,batch_size=32,num_steps=200,num_layer=3,learning_rate=1e-4):
	
	reset_graph()


	x=tf.placeholder(tf.int32,[None,num_steps],name='input')
	y=tf.placeholder(tf.int32,[None,num_steps],name='labels')

	embeddings=tf.get_variable(name='embedding_matrix',initializer=tf.random_normal([num_classes,state_size],mean=0.0,stddev=0.05))

	rnn_inputs=tf.nn.embedding_lookup(embeddings,x)

	cell=tf.nn.rnn_cell.LSTMCell(state_size,state_is_tuple=True)
	cell=tf.nn.rnn_cell.MultiRNNCell([cell]*num_layer,state_is_tuple=True)

	init_state=cell.zero_state(batch_size,tf.float32)
	rnn_outputs,final_state=tf.nn.dynamic_rnn(cell,rnn_inputs,initial_state=init_state)

	rnn_outputs=tf.reshape(rnn_outputs,[-1,state_size])
	y_reshaped=tf.reshape(y,[-1])
	

	with tf.variable_scope('softmax'):
		W=tf.get_variable('W',[state_size,num_classes])
		b=tf.get_variable('b',[num_classes],initializer=tf.constant_initializer(0.0))

	

	logits=tf.matmul(rnn_outputs,W)+b;
	predictions=tf.nn.softmax(logits)
	total_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped))
	train_step=tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
	saver=tf.train.Saver(max_to_keep=3,keep_checkpoint_every_n_hours=1)
	return dict(x=x,y=y,init_state=init_state,final_state=final_state,train_step=train_step,total_loss=total_loss,pred=predictions,saver=saver)

# start=time.time()
# build_graph()
# print 'Took, ', time.time()-start,' seconds to build graph'

def train(Graph,num_epochs=-1,num_steps=50,batch_size=32,verbose=True,save=True):
	tf.set_random_seed(2345)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		training_losses = []
		for idx,epoch in enumerate(gen_epochs(num_steps,batch_size)):
			training_loss=0
			steps=0
			X=epoch[0]
			Y=epoch[1]	
			steps+=1
			feed_dict={Graph['x']: X,Graph['y']: Y}
			temp_loss, training_state, _ = sess.run([Graph['total_loss'],Graph['final_state'],Graph['train_step']],feed_dict) 
			training_loss+=temp_loss
			if verbose:
				print 'Average training loss for epoch ', idx, ': ', training_loss/steps
			training_losses.append(training_loss/steps)
		if save:
			restore_from=Graph['saver'].save(sess,'language_model')
	return restore_from


def generate_characters(Graph,restoration_path=None,start='a',text_length=100,pick_top_chars=5):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if(restoration_path is not None):
			Graph['saver'].restore(sess,restoration_path)
		else:
			Graph['saver'].restore(sess,tf.train.latest_checkpoint('./'))
		state=None
		current_char=vocab_to_idx[start]
		chars=[current_char]
		# print tf.get_default_graph().get_tensor_by_name('embedding_matrix:0').eval()
		# print '######################################\n'
		for i in range(text_length):
			if(state is not None):
				feed_dict={Graph['x']: [[current_char]],Graph['init_state']: state}
			else:
				feed_dict={Graph['x']: [[current_char]]}

			pred, state=sess.run([Graph['pred'],Graph['final_state']],feed_dict)


			p=np.squeeze(pred)
			p[np.argsort(p)[:-pick_top_chars]]=0
			p=p/np.sum(p)
			current_char=np.random.choice(vocab_size,1,p=p)[0]
			chars.append(current_char)

	chars = map(lambda x: idx_to_vocab[x], chars)
	return chars

def check_mat():
	with tf.Session() as sess:
		saver=tf.train.Saver()
		saver.restore(sess,tf.train.latest_checkpoint('./'))
		graph=tf.get_default_graph()
		embeddings=graph.get_tensor_by_name('embedding_matrix:0')
		print embeddings.eval()



# start=time.time()
# g=build_graph(state_size=100,batch_size=32,num_steps=200)
# restoration_path=train(g,num_steps=200,batch_size=32)
# print 'Took, ', time.time()-start,' seconds to train'

g=build_graph(state_size=100,batch_size=1,num_steps=1)
# txt=generate_characters(g,restoration_path,text_length=1500)
txt=generate_characters(g,text_length=3500)
print "".join(txt)

# check_mat()





		


