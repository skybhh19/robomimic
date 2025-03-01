"""
Contains torch Modules that help deal with inputs consisting of multiple
modalities. This is extremely common when networks must deal with one or 
more observation dictionaries, where each input dictionary can have
observation keys of a certain modality and shape.

As an example, an observation could consist of a flat "robot0_eef_pos" observation key,
and a 3-channel RGB "agentview_image" observation key.
"""
import sys
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import Module, Sequential, MLP, RNN_Base, ResNet18Conv, SpatialSoftmax, \
    FeatureAggregator, VisualCore, Randomizer
from robomimic.models.feature_nets import SelfAttentionExtractor, MlpExtractor, SelfAttentionExtractor2, \
    DeepSetExtractor, GoalConditionedSelfAttentionExtractor

ATTENTION_FEATURES = None
MAX_NUM_OBJS = 4
NUM_PRIMITIVE_TYPE = 5
def obs_encoder_factory(
        obs_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.

    Args:
        obs_shapes (OrderedDict): a dictionary that maps observation key to
            expected shapes for observations.

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.

        encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should be
            nested dictionary containing relevant per-modality information for encoder networks.
            Should be of form:

            obs_modality1: dict
                feature_dimension: int
                core_class: str
                core_kwargs: dict
                    ...
                    ...
                obs_randomizer_class: str
                obs_randomizer_kwargs: dict
                    ...
                    ...
            obs_modality2: dict
                ...
    """
    enc = ObservationEncoder(feature_activation=feature_activation,
                             feature_extractor=encoder_kwargs["low_dim"]["feature_extractor"],
                             feature_kwargs=encoder_kwargs["low_dim"]["feature_kwargs"])
    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        enc_kwargs = deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality]) if encoder_kwargs is None else \
            deepcopy(encoder_kwargs[obs_modality])

        for obs_module, cls_mapping in zip(("core", "obs_randomizer"),
                                      (ObsUtils.OBS_ENCODER_CORES, ObsUtils.OBS_RANDOMIZERS)):
            # Sanity check for kwargs in case they don't exist / are None
            if enc_kwargs.get(f"{obs_module}_kwargs", None) is None:
                enc_kwargs[f"{obs_module}_kwargs"] = {}
            # Add in input shape info
            enc_kwargs[f"{obs_module}_kwargs"]["input_shape"] = obs_shape
            # If group class is specified, then make sure corresponding kwargs only contain relevant kwargs
            if enc_kwargs[f"{obs_module}_class"] is not None:
                enc_kwargs[f"{obs_module}_kwargs"] = extract_class_init_kwargs_from_dict(
                    cls=cls_mapping[enc_kwargs[f"{obs_module}_class"]],
                    dic=enc_kwargs[f"{obs_module}_kwargs"],
                    copy=False,
                )

        # Add in input shape info
        randomizer = None if enc_kwargs["obs_randomizer_class"] is None else \
            ObsUtils.OBS_RANDOMIZERS[enc_kwargs["obs_randomizer_class"]](**enc_kwargs["obs_randomizer_kwargs"])

        enc.register_obs_key(
            name=k,
            shape=obs_shape,
            net_class=enc_kwargs["core_class"],
            net_kwargs=enc_kwargs["core_kwargs"],
            randomizer=randomizer,
        )

    enc.make()
    return enc


class ObservationEncoder(Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with the encoder and then
    finally call @make to create the encoder networks. 
    """
    def __init__(self, feature_activation=nn.ReLU, feature_extractor=None, feature_kwargs=None):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation. 
        """
        super(ObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self.feature_extractor = feature_extractor
        self.feature_kwargs = feature_kwargs
        self._locked = False

    def register_obs_key(
        self, 
        name,
        shape, 
        net_class=None, 
        net_kwargs=None, 
        net=None, 
        randomizer=None,
        share_net_from=None,
    ):
        """
        Register an observation key that this encoder should be responsible for.

        Args:
            name (str): modality name
            shape (int tuple): shape of modality
            net_class (str): name of class in base_nets.py that should be used
                to process this observation key before concatenation. Pass None to flatten
                and concatenate the observation key directly.
            net_kwargs (dict): arguments to pass to @net_class
            net (Module instance): if provided, use this Module to process the observation key
                instead of creating a different net
            randomizer (Randomizer instance): if provided, use this Module to augment observation keys
                coming in to the encoder, and possibly augment the processed output as well
            share_net_from (str): if provided, use the same instance of @net_class 
                as another observation key. This observation key must already exist in this encoder.
                Warning: Note that this does not share the observation key randomizer
        """
        assert not self._locked, "ObservationEncoder: @register_obs_key called after @make"
        assert name not in self.obs_shapes, "ObservationEncoder: modality {} already exists".format(name)

        if net is not None:
            assert isinstance(net, Module), "ObservationEncoder: @net must be instance of Module class"
            assert (net_class is None) and (net_kwargs is None) and (share_net_from is None), \
                "ObservationEncoder: @net provided - ignore other net creation options"

        if share_net_from is not None:
            # share processing with another modality
            assert (net_class is None) and (net_kwargs is None)
            assert share_net_from in self.obs_shapes

        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        if randomizer is not None:
            assert isinstance(randomizer, Randomizer)
            if net_kwargs is not None:
                # update input shape to visual core
                net_kwargs["input_shape"] = randomizer.output_shape_in(shape)

        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = randomizer
        self.obs_share_mods[name] = share_net_from

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"
        print(self.feature_extractor)
        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[self.obs_nets_classes[k]](**self.obs_nets_kwargs[k])
            elif self.obs_share_mods[k] is not None:
                # make sure net is shared with another modality
                self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]
        if 'prior' in self.feature_extractor:
            robot_dim = self.obs_shapes['cur_robot0_eef_pos'][-1] + self.obs_shapes['cur_robot0_eef_quat'][-1] + self.obs_shapes['cur_robot0_gripper_qpos'][-1]
        elif 'policy' in self.feature_extractor:
            robot_dim = self.obs_shapes['robot0_eef_pos'][-1] + self.obs_shapes['robot0_eef_quat'][-1] + \
                        self.obs_shapes['robot0_gripper_qpos'][-1]
        object_dim = 7
        if self.feature_extractor in ['self_attention_prior', 'self_attention_policy', 'self_attention_policy_objectcentric',
                                      'self_attention_ll', 'self_attention_ll_objectcentric']:
            if 'objectcentric' in self.feature_extractor:
                robot_dim += 3
            self.feature_net = SelfAttentionExtractor(robot_dim, object_dim, hidden_size=self.feature_kwargs["hidden_size"],
                                                       n_attention_blocks=self.feature_kwargs["n_attention_blocks"],
                                                       n_heads=self.feature_kwargs["n_heads"])
            print("self_attention")
        elif self.feature_extractor in ['goal_self_attention_prior', 'goal_self_attention_prior_objectcentric']:
            if 'objectcentric' in self.feature_extractor:
                object_dim += 3
            self.feature_net = GoalConditionedSelfAttentionExtractor(robot_dim, object_dim,
                                                                     hidden_size=self.feature_kwargs["hidden_size"],
                                                                     n_attention_blocks=self.feature_kwargs["n_attention_blocks"],
                                                                     n_heads=self.feature_kwargs["n_heads"])
        elif self.feature_extractor == 'deepset_prior':
            self.feature_net = DeepSetExtractor(robot_dim, object_dim, hidden_size=self.feature_kwargs["hidden_size"])
        elif self.feature_extractor in ['self_attention2_prior', 'self_attention2_policy', 'self_attention2_policy_objectcentric',
                                        'self_attention2_ll', 'self_attention2_ll_objectcentric',
                                        'self_attention_policy_objectcentric']:
            if 'objectcentric' in self.feature_extractor:
                robot_dim += 3
            self.feature_net = SelfAttentionExtractor2(robot_dim, object_dim, hidden_size=self.feature_kwargs["hidden_size"],
                                                       n_attention_blocks=self.feature_kwargs["n_attention_blocks"],
                                                       n_heads=self.feature_kwargs["n_heads"], dropout_prob=0.0)
        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        # process modalities by order given by @self.obs_shapes
        feats = []
        if self.feature_extractor in ['self_attention_prior', 'goal_self_attention_prior', 'deepset_prior',
                                      'self_attention2_prior', 'goal_self_attention_prior_objectcentric']:
            feats.append(obs_dict['primitive_type'])
            cur_robot_obs = torch.cat([obs_dict['cur_robot0_eef_pos'], obs_dict['cur_robot0_eef_quat'], obs_dict['cur_robot0_gripper_qpos']], dim=-1)
            batch_size = obs_dict['cur_obj_pos'].shape[0]
            cur_object_obs = torch.cat([obs_dict['cur_obj_pos'].reshape(batch_size, -1, 3),
                                        obs_dict['cur_obj_quat'].reshape(batch_size, -1, 4),
                                        # obs_dict['cur_obj_ind'].reshape(batch_size, -1, MAX_NUM_OBJS)
                                        ], dim=-1)
            goal_robot_obs = torch.cat(
                [obs_dict['goal_robot0_eef_pos'], obs_dict['goal_robot0_eef_quat'], obs_dict['goal_robot0_gripper_qpos']],
                dim=-1)
            goal_object_obs = torch.cat([obs_dict['goal_obj_pos'].reshape(batch_size, -1, 3),
                                         obs_dict['goal_obj_quat'].reshape(batch_size, -1, 4),
                                         # obs_dict['goal_obj_ind'].reshape(batch_size, -1, MAX_NUM_OBJS)
                                         ], dim=-1)
            # if self.feature_extractor == 'mlp':
            #     cur_obs = torch.cat([cur_robot_obs, obs_dict['cur_obj_pos'], obs_dict['cur_obj_quat'], obs_dict['cur_object_centric']], dim=-1)
            #     goal_obs = torch.cat(
            #         [goal_robot_obs, obs_dict['goal_obj_pos'], obs_dict['goal_obj_quat'], obs_dict['goal_object_centric']],
            #         dim=-1)
            #     feats.extend([self.feature_net(cur_obs), self.feature_net(goal_obs)])
            # elif ATTENTION_FEATURES == 'self_attention':
            if self.feature_extractor in ['self_attention_prior', 'self_attention2_prior']:
                feats.append(self.feature_net1(cur_robot_obs, cur_object_obs))
                feats.append(self.feature_net2(goal_robot_obs, goal_object_obs))
            elif self.feature_extractor in ['goal_self_attention_prior', 'deepset_prior']:
                feats.append(self.feature_net(cur_robot_obs, cur_object_obs, goal_robot_obs, goal_object_obs))
            elif self.feature_extractor in ['goal_self_attention_prior_objectcentric']:
                cur_object_obs = torch.cat([cur_object_obs,
                                            obs_dict['cur_obj_pos'].reshape(batch_size, -1, 3)
                                            - obs_dict['cur_robot0_eef_pos'].reshape(batch_size, 1, 3)], dim=-1)
                goal_object_obs = torch.cat([goal_object_obs,
                                             obs_dict['goal_obj_pos'].reshape(batch_size, -1, 3)
                                             - obs_dict['goal_robot0_eef_pos'].reshape(batch_size, 1, 3)], dim=-1)
                feats.append(self.feature_net(cur_robot_obs, cur_object_obs, goal_robot_obs, goal_object_obs))
        elif self.feature_extractor in ['self_attention_policy', 'self_attention2_policy']:
            feats.append(obs_dict['primitive_type'])
            robot_obs = torch.cat(
                [obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'], obs_dict['robot0_gripper_qpos']],
                dim=-1)
            batch_size = obs_dict['obj_pos'].shape[0]
            object_obs = torch.cat([obs_dict['obj_pos'].reshape(batch_size, -1, 3),
                                    obs_dict['obj_quat'].reshape(batch_size, -1, 4),
                                    # obs_dict['obj_ind'].reshape(batch_size, -1, MAX_NUM_OBJS)
                                    ], dim=-1)
            feats.append(self.feature_net(robot_obs, object_obs))
        elif self.feature_extractor in ['self_attention_ll', 'self_attention2_ll']:
            robot_obs = torch.cat(
                [obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'], obs_dict['robot0_gripper_qpos']],
                dim=-1)
            batch_size = obs_dict['obj_pos'].shape[0]
            object_obs = torch.cat([obs_dict['obj_pos'].reshape(batch_size, -1, 3),
                                    obs_dict['obj_quat'].reshape(batch_size, -1, 4),
                                    # obs_dict['obj_ind'].reshape(batch_size, -1, MAX_NUM_OBJS)
                                    ], dim=-1)
            feats.append(self.feature_net(robot_obs, object_obs))
        elif self.feature_extractor in ['self_attention_policy_objectcentric', 'self_attention2_policy_objectcentric']:
            feats.append(obs_dict['primitive_type'])
            robot_obs = torch.cat(
                [obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'], obs_dict['robot0_gripper_qpos']],
                dim=-1)
            batch_size = obs_dict['obj_pos'].shape[0]
            object_obs = torch.cat([obs_dict['obj_pos'].reshape(batch_size, -1, 3),
                                    obs_dict['obj_quat'].reshape(batch_size, -1, 4),
                                    # obs_dict['obj_ind'].reshape(batch_size, -1, MAX_NUM_OBJS),
                                    obs_dict['obj_pos'].reshape(batch_size, -1, 3) - obs_dict['robot0_eef_pos'].reshape(batch_size, 1, 3)], dim=-1)
            feats.append(self.feature_net(robot_obs, object_obs))
        elif self.feature_extractor in ['self_attention_ll_objectcentric', 'self_attention2_ll_objectcentric']:
            robot_obs = torch.cat(
                [obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'], obs_dict['robot0_gripper_qpos']],
                dim=-1)
            batch_size = obs_dict['obj_pos'].shape[0]
            object_obs = torch.cat([obs_dict['obj_pos'].reshape(batch_size, -1, 3),
                                    obs_dict['obj_quat'].reshape(batch_size, -1, 4),
                                    # obs_dict['obj_ind'].reshape(batch_size, -1, MAX_NUM_OBJS),
                                    obs_dict['obj_pos'].reshape(batch_size, -1, 3) - obs_dict['robot0_eef_pos'].reshape(batch_size, 1, 3)], dim=-1)
            feats.append(self.feature_net(robot_obs, object_obs))
        elif self.feature_extractor == 'mlp':
            for k in self.obs_shapes:
                x = obs_dict[k]
                # maybe process encoder input with randomizer
                if self.obs_randomizers[k] is not None:
                    x = self.obs_randomizers[k].forward_in(x)
                # maybe process with obs net
                if self.obs_nets[k] is not None:
                    x = self.obs_nets[k](x)
                    if self.activation is not None:
                        x = self.activation(x)
                # maybe process encoder output with randomizer
                if self.obs_randomizers[k] is not None:
                    x = self.obs_randomizers[k].forward_out(x)
                # flatten to [B, D]
                x = TensorUtils.flatten(x, begin_axis=1)
                feats.append(x)
        else:
            raise NotImplementedError
        # concatenate all features together
        return torch.cat(feats, dim=-1)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        if self.feature_extractor in ['self_attention_prior', 'self_attention2_prior']:
            return [self.feature_net.hidden_size * 2 + NUM_PRIMITIVE_TYPE]
        elif self.feature_extractor in ['goal_self_attention_prior', 'goal_self_attention_prior_objectcentric',
                                        'deepset_prior', 'self_attention_policy', 'self_attention2_policy',
                                        'self_attention2_policy_objectcentric', 'self_attention_policy_objectcentric']:
            return [self.feature_net.hidden_size + NUM_PRIMITIVE_TYPE]
        elif self.feature_extractor in ['self_attention_ll', 'self_attention2_ll', 'self_attention_ll_objectcentric',
                                        'self_attention2_ll_objectcentric']:
            return [self.feature_net.hidden_size]
        elif self.feature_extractor == 'mlp':
            feat_dim = 0
            for k in self.obs_shapes:
                feat_shape = self.obs_shapes[k]
                if self.obs_randomizers[k] is not None:
                    feat_shape = self.obs_randomizers[k].output_shape_in(feat_shape)
                if self.obs_nets[k] is not None:
                    feat_shape = self.obs_nets[k].output_shape(feat_shape)
                if self.obs_randomizers[k] is not None:
                    feat_shape = self.obs_randomizers[k].output_shape_out(feat_shape)
                feat_dim += int(np.prod(feat_shape))
            return [feat_dim]
        else:
            raise NotImplementedError

    def __repr__(self):
        """
        Pretty print the encoder.
        """
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("randomizer={}\n".format(self.obs_randomizers[k]), indent)
            msg += textwrap.indent("net={}\n".format(self.obs_nets[k]), indent)
            msg += textwrap.indent("sharing_from={}\n".format(self.obs_share_mods[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    """
    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.obs_shapes[k]) for k in self.obs_shapes }

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class ObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and 
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """
    def __init__(
        self,
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(ObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all([isinstance(observation_group_shapes[k], OrderedDict) for k in observation_group_shapes])
        
        self.observation_group_shapes = observation_group_shapes

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
            )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with 
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(inputs), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(
                self.nets[obs_group].forward(inputs[obs_group])
            )

        return torch.cat(outputs, dim=-1)

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.observation_group_shapes:
            msg += '\n'
            indent = ' ' * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class MIMO_MLP(Module):
    """
    Extension to MLP to accept multiple observation dictionaries as input and
    to output dictionaries of tensors. Inputs are specified as a dictionary of 
    observation dictionaries, with each key corresponding to an observation group.

    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating 
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        layer_dims,
        layer_func=nn.Linear, 
        activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            layer_func: mapping per MLP layer - defaults to Linear

            activation: non-linearity per MLP layer - defaults to ReLU

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(MIMO_MLP, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        mlp_input_dim = self.nets["encoder"].output_shape()[0]

        # intermediate MLP layers
        self.nets["mlp"] = MLP(
            input_dim=mlp_input_dim,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation, # make sure non-linearity is applied before decoder
        )
        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=layer_dims[-1],
        )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes.

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes
        """
        enc_outputs = self.nets["encoder"](**inputs)
        mlp_out = self.nets["mlp"](enc_outputs)
        return self.nets["decoder"](mlp_out)

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class RNN_MIMO_MLP(Module):
    """
    A wrapper class for a multi-step RNN and a per-step MLP and a decoder.

    Structure: [encoder -> rnn -> mlp -> decoder]

    All temporal inputs are processed by a shared @ObservationGroupEncoder,
    followed by an RNN, and then a per-step multi-output MLP. 
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        mlp_activation=nn.ReLU,
        mlp_layer_func=nn.Linear,
        per_step=True,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the rnn model

            per_step (bool): if True, apply the MLP and observation decoder into @output_shapes
                at every step of the RNN. Otherwise, apply them to the final hidden state of the 
                RNN.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(RNN_MIMO_MLP, self).__init__()
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.per_step = per_step

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        rnn_input_dim = self.nets["encoder"].output_shape()[0]

        # bidirectional RNNs mean that the output of RNN will be twice the hidden dimension
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)
        num_directions = int(rnn_is_bidirectional) + 1 # 2 if bidirectional, 1 otherwise
        rnn_output_dim = num_directions * rnn_hidden_dim

        per_step_net = None
        self._has_mlp = (len(mlp_layer_dims) > 0)
        if self._has_mlp:
            self.nets["mlp"] = MLP(
                input_dim=rnn_output_dim,
                output_dim=mlp_layer_dims[-1],
                layer_dims=mlp_layer_dims[:-1],
                output_activation=mlp_activation,
                layer_func=mlp_layer_func
            )
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=mlp_layer_dims[-1],
            )
            if self.per_step:
                per_step_net = Sequential(self.nets["mlp"], self.nets["decoder"])
        else:
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=rnn_output_dim,
            )
            if self.per_step:
                per_step_net = self.nets["decoder"]

        # core network
        self.nets["rnn"] = RNN_Base(
            input_dim=rnn_input_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            per_step_net=per_step_net,
            rnn_kwargs=rnn_kwargs
        )

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)

        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        return self.nets["rnn"].get_rnn_init_state(batch_size, device=device)

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.

        Args:
            input_shape (dict): dictionary of dictionaries, where each top-level key
                corresponds to an observation group, and the low-level dictionaries
                specify the shape for each modality in an observation dictionary
        """

        # infers temporal dimension from input shape
        obs_group = list(self.input_obs_group_shapes.keys())[0]
        mod = list(self.input_obs_group_shapes[obs_group].keys())[0]
        T = input_shape[obs_group][mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="RNN_MIMO_MLP: input_shape inconsistent in temporal dimension")
        # returns a dictionary instead of list since outputs are dictionaries
        return { k : [T] + list(self.output_shapes[k]) for k in self.output_shapes }

    def forward(self, rnn_init_state=None, return_state=False, **inputs):
        """
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.

            rnn_state (torch.Tensor or tuple): return the new rnn state (if @return_state)
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        # use encoder to extract flat rnn inputs
        rnn_inputs = TensorUtils.time_distributed(inputs, self.nets["encoder"], inputs_as_kwargs=True)
        assert rnn_inputs.ndim == 3  # [B, T, D]
        if self.per_step:
            return self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
        
        # apply MLP + decoder to last RNN output
        outputs = self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
        if return_state:
            outputs, rnn_state = outputs

        assert outputs.ndim == 3 # [B, T, D]
        if self._has_mlp:
            outputs = self.nets["decoder"](self.nets["mlp"](outputs[:, -1]))
        else:
            outputs = self.nets["decoder"](outputs[:, -1])

        if return_state:
            return outputs, rnn_state
        return outputs

    def forward_step(self, rnn_state, **inputs):
        """
        Unroll network over a single timestep.

        Args:
            inputs (dict): expects same modalities as @self.input_shapes, with
                additional batch dimension (but NOT time), since this is a 
                single time step.

            rnn_state (torch.Tensor): rnn hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Does not contain time dimension.

            rnn_state: return the new rnn state
        """
        # ensure that the only extra dimension is batch dim, not temporal dim 
        assert np.all([inputs[k].ndim - 1 == len(self.input_shapes[k]) for k in self.input_shapes])

        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(
            inputs, 
            rnn_init_state=rnn_state,
            return_state=True,
        )
        if self.per_step:
            # if outputs are not per-step, the time dimension is already reduced
            outputs = outputs[:, 0]
        return outputs, rnn_state

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\n\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nrnn={}".format(self.nets["rnn"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg
