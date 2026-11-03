"""Published PAP Keras inference with backend-import adaptation and NumPy check."""
from pathlib import Path
import ast
import os
import sys
import numpy as np
import h5py

P=Path(__file__).resolve().parent
O=P/'pap_neural_comparison'
sys.path.insert(0,str(O/'vendor'))
os.environ['KERAS_BACKEND']='torch'
os.environ['KERAS_HOME']=str(P.parent/'tmp/pap_keras_home')
os.environ['TORCH_HOME']=str(P.parent/'tmp/pap_torch_home')
import keras
import torch
torch.set_num_threads(4)
from freeze_published_baselines import literal_model

AA='ACDEFGHIKLMNPQRSTVWY'


class Tile:
    def __init__(self,seq):
        self.seq=seq
    def __len__(self):
        return len(self.seq)
    def is_ct(self):
        return True


def published_classes():
    source=ast.parse((O/'PAP.py').read_text(encoding='utf-8'))
    nodes=[n for n in source.body if isinstance(n,ast.ClassDef) and n.name in ['KerasModel','KerasModelCNN2W']]
    assert len(nodes)==2
    class BackendImport(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module=='tensorflow':
                assert len(node.names)==1 and node.names[0].name=='keras'
                return ast.copy_location(ast.Import(names=[ast.alias(name='keras')]),node)
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='load_weights':
                # Keras3's torch backend treats the original trainable=False
                # tensors as top-level weights and rejects this legacy HDF5.
                # Keep the published graph and load all eight named layer
                # tensors explicitly, with exhaustive shape/value checks.
                return ast.copy_location(ast.Call(func=ast.Name(id='_load_named_weights',ctx=ast.Load()),
                    args=[node.func.value]+node.args,keywords=node.keywords),node)
            return node
    module=ast.Module(body=nodes,type_ignores=[])
    module=ast.fix_missing_locations(BackendImport().visit(module))
    namespace=dict(np=np,os=os,sys=sys,_load_named_weights=load_named_weights)
    exec(compile(module,str(O/'PAP.py')+' [backend/legacy-loader adaptation]','exec'),namespace)
    return namespace['KerasModelCNN2W']


def load_named_weights(model,file,**kwargs):
    loaded=[]
    with h5py.File(file,'r') as h:
        for layer in model.layers:
            if not layer.weights:
                continue
            arrays=[h[f'{layer.name}/{layer.name}/kernel:0'][()],h[f'{layer.name}/{layer.name}/bias:0'][()]]
            assert len(layer.get_weights())==2
            assert all(a.shape==b.shape for a,b in zip(arrays,layer.get_weights()))
            layer.set_weights(arrays)
            assert all(np.array_equal(a,b) for a,b in zip(arrays,layer.get_weights()))
            loaded.append(layer.name)
    assert set(loaded)=={'int_cnn','int2','int3','latent_int','ct1','ct2','ct3','latent_ct'}


class PublishedPAP:
    def __init__(self,name):
        self.name=name
        self.param=literal_model(O/'PAP.py',name)
        self.path=O/'pap_weights'/f'{name}_weights.h5'
        self.param['weight_file']=str(self.path.resolve())
        self.model=published_classes()(self.param)
        self.weights={}
        with h5py.File(self.path,'r') as h:
            for layer in ['int_cnn','int2','int3','latent_int','ct1','ct2','ct3','latent_ct']:
                arrays=[h[f'{layer}/{layer}/kernel:0'][()],h[f'{layer}/{layer}/bias:0'][()]]
                self.weights[layer]=[x.astype(np.float64) for x in arrays]
                branch=self.model.model_int if layer.startswith(('int','latent_int')) else self.model.model_ct
                loaded=branch.get_layer(layer).get_weights()
                assert all(np.array_equal(a,b) for a,b in zip(arrays,loaded)),layer

    def predict(self,seqs,batch_size=4096):
        assert len({len(s) for s in seqs})==1
        n=len(seqs[0])
        assert n>=self.param['cnn_res'] and n>=self.param['ct_res']
        result=[]
        for start in range(0,len(seqs),batch_size):
            batch=seqs[start:start+batch_size]
            full,ct=self.model.predict([Tile(s) for s in batch],include_ct=True)
            result.append(np.column_stack([full,np.array(full)-np.array(ct),ct]))
        return np.concatenate(result,axis=0)

    def internal(self,seqs,batch_size=4096):
        result=[]
        for start in range(0,len(seqs),batch_size):
            x=self.model.seqs2mat(seqs[start:start+batch_size]).astype('float32')
            result.append(self.model.model_int.predict(x,batch_size=512,verbose=0).flatten())
        return np.concatenate(result)

    def numpy_predict(self,seqs):
        assert len({len(s) for s in seqs})==1
        index={a:i for i,a in enumerate(AA)}
        x=np.eye(20)[np.array([[index[a] for a in s] for s in seqs])]
        k=self.param['cnn_res']
        w,b=self.weights['int_cnn']
        out=np.zeros((len(seqs),x.shape[1]-k+1,w.shape[2]))
        for j in range(k):
            out+=x[:,j:j+out.shape[1],:]@w[j]
        def elu(v):
            v=v.copy(); mask=v<0; v[mask]=np.expm1(v[mask]); return v
        out=elu(out+b).mean(axis=1)
        for layer in ['int2','int3','latent_int']:
            w,b=self.weights[layer]
            out=out@w+b
            if layer!='latent_int':
                out=elu(out)
        ct=x[:,-self.param['ct_res']:,:].reshape(len(seqs),-1)
        for layer in ['ct1','ct2','ct3','latent_ct']:
            w,b=self.weights[layer]
            ct=ct@w+b
            if layer!='latent_ct':
                ct=elu(ct)
        return np.column_stack([(out+ct).ravel(),out.ravel(),ct.ravel()])
