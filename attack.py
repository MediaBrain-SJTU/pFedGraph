import torch
import numpy as np

def manipulate_gradient(args, global_model, nets_this_round, benign_client_list, indivial_init_model=None):
    for client_id, net in nets_this_round.items():
        if client_id not in benign_client_list:
            manipulate_one_model(args, net, client_id, global_model, indivial_init_model)

def manipulate_one_model(args, net, client_id, global_model=None, indivial_init_model=None):
    print(f'Manipulating Client {client_id}')
    if args.attack_type == 'inv_grad':       # inverse the gradient of client
        start_w = indivial_init_model[client_id].state_dict() if indivial_init_model is not None else global_model.state_dict()
        local_w = net.state_dict()
        local_w = inverse_gradient(start_w, local_w)
        net.load_state_dict(local_w)
    elif args.attack_type == 'shuffle':     # shuffle model parameters
        flat_params = get_flat_params_from(net)
        shuffled_flat_params = flat_params[torch.randperm(len(flat_params))]
        set_flat_params_to(net, shuffled_flat_params)
    elif args.attack_type == 'same_value':
        flat_params = get_flat_params_from(net)
        flat_params = torch.ones_like(flat_params)
        set_flat_params_to(net, flat_params)
    elif args.attack_type == 'sign_flip':
        flat_params = get_flat_params_from(net)
        flat_params = -flat_params
        set_flat_params_to(net, flat_params)
    elif args.attack_type == 'gauss':
        flat_params = get_flat_params_from(net)
        flat_params = torch.normal(0, 1, size=flat_params.shape)
        set_flat_params_to(net, flat_params)

            
def inverse_gradient(global_w, local_w):
    for key in local_w:
        local_w[key] = global_w[key] - (local_w[key] - global_w[key])
    return local_w

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size