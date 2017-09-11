from __future__ import generators
import tensorflow as tf
import time
import numpy as np


f='code.txt'
vocab=set()
vocab_size=0
vocab_to_idx=dict()
idx_to_vocab=list()
def get_encoding(f):
	switch=False
	i=0
	with open(f,'r') as F:
	    for line in F:
	    	line=line[1:len(line)-1]
	    	line=line.split(',')
	    	for x in line:
	    		print x


get_encoding(f)         


'''
with open('format_allines.txt','r') as F:
	raw_data=F.read()
	data=[vocab_to_idx[c.lower()] for c in raw_data]

print(vocab_size)
print(len(data))
del raw_data


def reset_graph():
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()


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




g=build_graph(state_size=100,batch_size=1,num_steps=1)
txt=generate_characters(g,text_length=3500)
print("".join(txt))
'''


