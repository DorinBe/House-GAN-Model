import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn
import torch
from models.models import Generator
from misc.utils import _init_input, draw_masks, draw_graph
import io
import flask
from flask import send_file
import json
from data.data import myjsonpath, mytextpath
from google.cloud import storage
from dump.dump import dump_path
from zipfile import ZipFile

app = flask.Flask(__name__)
def get_from_cloud():
    """Model is saved in cloud to save space in the cotainer. This function downloads the model from the cloud."""
    storage_client = storage.Client(project="house-gan-model")
    bucket_name = "checkpoints-12"
    blob_name = "pretrained.pth"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return content

def _infer(graph, model, prev_state=None):
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
    with torch.no_grad():
        masks = model(z.to('cpu'), given_masks_in.to('cpu'), given_nds.to('cpu'), given_eds.to('cpu'))
        masks = masks.detach().cpu().numpy()
    return masks

@app.route('/generate', methods=['POST'])
def generate():
    my_data = flask.request.json
    with open(myjsonpath, 'w') as json_file:
        json.dump(my_data, json_file) # model have to read from json file
    # create txt file
    with open(mytextpath, 'w') as f:
        f.write(myjsonpath+'\n') # model have to read from txt file the jsons

    content = get_from_cloud()
    buffer = io.BytesIO(content)
    model = Generator()
    model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')), strict=True)
    model = model.eval()

    fp_dataset_test = FloorplanGraphDataset(mytextpath, transforms.Normalize(mean=[0.5], std=[0.5]), split='test')
    fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                            batch_size=1, 
                                            shuffle=False, collate_fn=floorplan_collate_fn)

    globalIndex = 0
    for i, sample in enumerate(fp_loader):
        # draw real graph and groundtruth
        mks, nds, eds, _, _ = sample
        real_nodes = np.where(nds.detach().cpu()==1)[-1]
        graph = [nds, eds]
        true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
        graph_path = os.path.join(dump_path, f'graph_{i}.png')
        graph_im.save(graph_path)

        # add room types incrementally
        _types = sorted(list(set(real_nodes)))
        selected_types = [_types[:k+1] for k in range(10)]
        # os.makedirs('./{}/'.format(dump_path), exist_ok=True) # for the mean time
        _round = 0
        
        # initialize layout
        state = {'masks': None, 'fixed_nodes': []}
        masks = _infer(graph, model, state)
        im0 = draw_masks(masks.copy(), real_nodes)
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0

        # generate per room type
        for _iter, _types in enumerate(selected_types):
            _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
                if len(_types) > 0 else np.array([]) 
            state = {'masks': masks, 'fixed_nodes': _fixed_nds}
            masks = _infer(graph, model, state)
            
        # save final floorplans
        imk = draw_masks(masks.copy(), real_nodes)
        imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
        fp_img_io = io.BytesIO()
        save_image(imk, fp_img_io, format='PNG', nrow=1, normalize=False)
        image_path = os.path.join(dump_path, f"fp_{i}.png")
        save_image(imk, image_path, "PNG")
        fp_img_io.seek(0)
        
        zip_path = os.path.join(dump_path, f"output.zip")
        with ZipFile(zip_path, 'w') as zipf:
            zipf.write(image_path)
            zipf.write(graph_path)

        return send_file(zip_path, mimetype='application/zip', as_attachment=True)


app.run(port=int(os.environ.get('PORT', 8080)), host='0.0.0.0',debug=True)