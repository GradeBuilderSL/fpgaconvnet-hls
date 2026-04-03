import os
import sys
from functools import reduce
from pathlib import Path

import onnx
import numpy as np
from google.protobuf.json_format import MessageToDict

from fpgaconvnet.hls.generate.partition_template import *

from fpgaconvnet.hls.generate.layers.convolution        import gen_convolution_layer
from fpgaconvnet.hls.generate.layers.pooling            import gen_pooling_layer
from fpgaconvnet.hls.generate.layers.avg_pooling        import gen_avg_pooling_layer
from fpgaconvnet.hls.generate.layers.global_pooling     import gen_global_pooling_layer
from fpgaconvnet.hls.generate.layers.relu               import gen_relu_layer
from fpgaconvnet.hls.generate.layers.inner_product      import gen_inner_product_layer
from fpgaconvnet.hls.generate.layers.squeeze            import gen_squeeze_layer
from fpgaconvnet.hls.generate.layers.split              import gen_split_layer
from fpgaconvnet.hls.generate.layers.elementwise_add    import gen_elementwise_add_layer
from fpgaconvnet.hls.generate.layers.elementwise_mul    import gen_elementwise_mul_layer
from fpgaconvnet.hls.generate.util import *

import fpgaconvnet.hls.tools.onnx_data as onnx_data
from fpgaconvnet.hls.tools.array_init import array_init

import fpgaconvnet.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet.parser import Parser
import fpgaconvnet.parser.onnx.helper as onnx_helper
import fpgaconvnet.tools.layer_enum as layer_enum

class GeneratePartition:

    def __init__(self, name, partition, model, sess, output_path, port_width=64):

        self.name = name
        self.partition = partition
        self.output_path = output_path
        self.model = model
        self.sess = sess
        self.port_width = port_width

        # make output path directory
        self.mkdir(self.output_path)
        self.mkdir(os.path.join(self.output_path, "src"))
        self.mkdir(os.path.join(self.output_path, "tb"))
        self.mkdir(os.path.join(self.output_path, "include"))
        self.mkdir(os.path.join(self.output_path, "data"))

        # get fpgaconvnet-hls root directory
        self.fpgaconvnet_root = str(Path(__file__).resolve().parent.parent)

        # state of generated partition
        self.is_generated = {
            "layers"    : False,
            "weights"   : False,
            "streams"   : False,
            "source"    : False,
            "include"   : False,
            "testbench" : False,
        }
        self.project_generated = False

    def mkdir(self, path):
        """
        Helper function to create a directory, which does not throw an error
        if the path already exists.

        Parameters
        ----------
        path: str
            path to directory to create
        """
        try:
            os.mkdir(path)
        except FileExistsError:
            print(f"WARNING: path {path} already exists!")

    def generate_layers(self):
        """
        Generates layer header and source files through the layer generator functions,
        and creates a string of layer calls for
        """

        self.layers = ""

        # generate hardware
        for layer in self.partition.layers:
            # get parameters of layer
            parameters = MessageToDict(layer.parameters, preserving_proto_field_name=True)
            # FIXME adding buffer depth to params in a hacky way
            parameters['buffer_depth'] = layer.streams_in[0].buffer_depth
            # init function arguments for this layer
            fn_args = []
            # init hardware generation args
            args = [
                layer.name,
                parameters,
                os.path.join(self.output_path, "src", f"{layer.name}.cpp"),
                os.path.join(self.output_path, "include", f"{layer.name}.hpp")
            ]
            # create hardware for each layer
            print(f"--------- Generating layer {layer.name} ---------")
            if layer.type == fpgaconvnet_pb2.layer.layer_type.CONVOLUTION:
                fn_args.append(f"{layer.name}_weights")
                if layer.parameters.has_bias == 1:
                    is_wr_layer = (layer.name == self.partition.weights_reloading_layer)
                    if is_wr_layer and self.partition.weights_reloading_factor > 1:
                        fn_args.append(f"{layer.name}_biases[weights_reloading_index]")
                    else:
                        fn_args.append(f"{layer.name}_biases")
                gen_convolution_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.POOLING:
                if layer.name.startswith("Max"):
                    gen_pooling_layer(*args)
                elif layer.name.startswith("Average"):
                    gen_avg_pooling_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.AVERAGE_POOLING:
                gen_global_pooling_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.CONCAT:
                gen_concat_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.RELU:
                gen_relu_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.SPLIT:
                gen_split_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT:
                fn_args.append(f"{layer.name}_weights")
                if layer.parameters.has_bias == 1:
                    is_wr_layer = (layer.name == self.partition.weights_reloading_layer)
                    if is_wr_layer and self.partition.weights_reloading_factor > 1:
                        fn_args.append(f"{layer.name}_biases[weights_reloading_index]")
                    else:
                        fn_args.append(f"{layer.name}_biases")
                gen_inner_product_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.SQUEEZE:
                gen_squeeze_layer(*args)
            if layer.type == fpgaconvnet_pb2.layer.layer_type.ELTWISE:
                if layer.name.startswith("Add"):
                    gen_elementwise_add_layer(*args)
                elif layer.name.startswith("Mul"):
                    gen_elementwise_mul_layer(*args)
                else: raise ValueError("Operation type can only be Add or Mul")
            # create layer call
            for stream_in in layer.streams_in:
                fn_args.append(stream_in.name)
            for stream_out in layer.streams_out:
                fn_args.append(stream_out.name)
            fn_args.append("mode")
            fn_args = ", ".join(fn_args)
            self.layers += f"    {layer.name}({fn_args});\n"

        # set generated flag
        self.is_generated["layers"] = True

    def generate_parameters(self):

        weights = []
        biases = []

        # generate weights
        for layer in self.partition.layers:
            # get parameters of layer
            parameters = MessageToDict(layer.parameters, preserving_proto_field_name=True)
            # create hardware for each layer
            if layer.type in [fpgaconvnet_pb2.layer.layer_type.CONVOLUTION,
                    fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT]:
                # add weight generators
                weights.append(GenerateWeights(layer.name,
                        wr=(layer.name == self.partition.weights_reloading_layer)))
                # create weights from onnx model
                ## get weights reloading factor
                wr_factor = self.partition.weights_reloading_factor \
                        if layer.name == self.partition.weights_reloading_layer else 1
                ## get the raw weights
                weights_raw = onnx_helper.get_model_initializer(self.model, layer.weights_path)
                ## transforms weights
                if layer_enum.from_proto_layer_type(layer.type) == layer_enum.LAYER_TYPE.Convolution:
                    transformed_weights = onnx_data.get_weights_convolution(weights_raw, layer, wr_factor=wr_factor)
                elif layer_enum.from_proto_layer_type(layer.type) == layer_enum.LAYER_TYPE.InnerProduct:
                    #TODO: Support transA and transB parameters
                    weights_raw = weights_raw.T
                    transformed_weights = onnx_data.get_weights_inner_product(weights_raw, layer, wr_factor=wr_factor)
                ## get the output path for the weights
                output_path = os.path.join(self.output_path, "data", f"{layer.name}_weights")
                ## save weights to csv
                for weights_reloading_index in range(wr_factor):
                    with open(f'{output_path}_{weights_reloading_index}.csv', 'w') as f:
                        f.write(array_init(transformed_weights[weights_reloading_index]))
                ## flatten weights into a stream
                weight_int_width = layer.parameters.weight_t.width - \
                    layer.parameters.weight_t.binary_point
                weights_stream =  onnx_data._convert_fixed_port_stream(
                        transformed_weights.reshape(-1),
                        total_width=layer.parameters.weight_t.width,
                        int_width=weight_int_width)
                ## save to .dat format
                print(f"Writing weights stream to .dat file")
                onnx_data._fixed_point_stream_to_dat(weights_stream, output_path=output_path,
                        streams=1, port_width=self.port_width, ports=1)
                # add weight generators (if bias present)
                if layer.parameters.has_bias == 1:
                    ## add a biases generator (pass wr_factor so declaration gets extra dimension for WR layers)
                    biases.append(GenerateBiases(layer.name, wr_factor=wr_factor))
                    # create bias parameters from onnx model
                    ## get the raw biases from onnx
                    biases_raw = onnx_helper.get_model_initializer(self.model, layer.bias_path)
                    ## transform bias parameters
                    transformed_biases = onnx_data.get_biases(biases_raw, layer, wr_factor=wr_factor)
                    ## get the output path for biases
                    output_path = os.path.join(self.output_path, "data", f"{layer.name}_biases")
                    ## save biases to csv
                    with open(f'{output_path}.csv', 'w') as f:
                        if wr_factor > 1:
                            # write all WR groups so hardware can index by weights_reloading_index
                            f.write(array_init(transformed_biases))
                        else:
                            f.write(array_init(transformed_biases[0]))
                    ## flatten biases into a stream
                    # FIXME check if bias width should be accum width or smth else
                    acc_int_width = layer.parameters.acc_t.width - \
                        layer.parameters.acc_t.binary_point
                    biases_stream = onnx_data._convert_fixed_port_stream(
                            transformed_biases.reshape(-1),
                            total_width=layer.parameters.acc_t.width,
                            int_width=acc_int_width)
                    ## save to .dat format
                    # onnx_data._fixed_point_stream_to_dat(biases_stream, output_path=output_path,
                    #         streams=1, port_width=64, ports=1)
                    print("Writing biases stream to .dat file")
                    onnx_data._fixed_point_stream_to_dat(biases_stream, output_path=output_path,
                            streams=1, port_width=self.port_width, ports=1)

        # get weights definitions and intialisation
        self.weights_def = "\n\n".join([w.generate_def() for w in weights])
        self.weights_init = "\n\n".join([w.generate_init() for w in weights])

        self.biases_def = "\n\n".join([b.generate_def() for b in biases])
        self.biases_init = "\n\n".join([b.generate_init() for b in biases])

        # generate process() parameter declarations and call arguments for weights/biases
        all_params = ([w.generate_process_param() for w in weights] +
                      [b.generate_process_param() for b in biases])
        if all_params:
            self.weights_process_params = ",\n    " + ",\n    ".join(all_params)
            self.weights_process_args   = ", " + ", ".join(
                [f"{w.name}_weights" for w in weights] +
                [f"{b.name}_biases"  for b in biases])
        else:
            self.weights_process_params = ""
            self.weights_process_args   = ""

        # set generated flag
        self.is_generated["weights"] = True

    def generate_streams(self):
        # get all streams
        streams = {}
        for layer in self.partition.layers:
            for stream_in in layer.streams_in:
                streams[stream_in.name] = GenerateStreams(stream_in.name, f"{layer.name}_input_t",
                        [f"{layer.name.upper()}_COARSE_IN"])
            for stream_out in layer.streams_out:
                streams[stream_out.name] = GenerateStreams(stream_out.name, f"{layer.name}_output_t",
                        [f"{layer.name.upper()}_COARSE_OUT"])

        # create stream initialisations
        self.streams_init = "\n".join([s.generate_stream() for s in streams.values()])

        # set generated flag
        self.is_generated["streams"] = True

    def generate_include(self):
        # include generation
        include = ""
        for layer in self.partition.layers:
            include +=f"#include \"{layer.name}.hpp\"\n"

        # get input, weight and output data width
        if (self.partition.layers[0].parameters.input_t.width != 0):
            input_data_width = self.partition.layers[0].parameters.input_t.width
        elif (self.partition.layers[0].parameters.data_t.width != 0):
            input_data_width = self.partition.layers[0].parameters.data_t.width
        else:
            raise ValueError("Input data width not found")
        
        if (self.partition.weights_reloading_layer != "None"):
            for layer in self.partition.layers:
                if (layer.name == self.partition.weights_reloading_layer):
                    if (layer.parameters.weight_t.width != 0):
                        weight_data_width = layer.parameters.weight_t.width
                    else:
                        raise ValueError("Weight data width not found")
        else: 
            weight_data_width = 0
        
        if (self.partition.layers[-1].parameters.output_t.width != 0):
            output_data_width = self.partition.layers[-1].parameters.output_t.width
        elif (self.partition.layers[-1].parameters.data_t.width != 0):
            output_data_width = self.partition.layers[-1].parameters.data_t.width
        else:
            raise ValueError("Output data width not found")

        # HEADER
        network_header = network_header_template.format(
            name                  =self.name,
            NAME                  =self.name.upper(),
            weights_process_params=self.weights_process_params,
            batch_size  =self.partition.batch_size,
            rows_in     =self.partition.layers[0].parameters.rows_in,
            cols_in     =self.partition.layers[0].parameters.cols_in,
            channels_in =self.partition.layers[0].parameters.channels_in,
            rows_out    =self.partition.layers[-1].parameters.rows_out,
            cols_out    =self.partition.layers[-1].parameters.cols_out,
            channels_out=self.partition.layers[-1].parameters.channels_out,
            ports       =self.partition.ports,
            streams_in  =self.partition.layers[0].parameters.coarse_in,
            streams_out =self.partition.layers[-1].parameters.coarse_out,
            input_layer =self.partition.layers[0].name,
            output_layer=self.partition.layers[-1].name,
            wr_layer    =self.partition.weights_reloading_layer,
            WR_LAYER    =self.partition.weights_reloading_layer.upper(),
            wr_factor   =self.partition.weights_reloading_factor,
            wr_flag     =int(self.partition.weights_reloading_layer != "None"),
            DMA_WIDTH   =self.port_width,
            input_data_width=input_data_width,
            weight_data_width=weight_data_width,
            output_data_width=output_data_width,
            include     =include
        )

        # save to output path
        with open(os.path.join(self.output_path, f'include/{self.name}_top.hpp'),'w') as f:
            f.write(network_header)

        # set generated flag
        self.is_generated["include"] = True

    def generate_source(self):

        # todo: check for layers, weights, streams
        assert self.is_generated["layers"], "ERROR: layers not generated!"
        assert self.is_generated["weights"], "ERROR: weights not generated!"
        assert self.is_generated["streams"], "ERROR: weights not generated!"

        # format the source code template
        network_src = network_src_template.format(
            name                  =self.name,
            NAME                  =self.name.upper(),
            wr_layer              =self.partition.weights_reloading_layer,
            weights               =self.weights_def,
            biases                =self.biases_def,
            weights_init          =self.weights_init,
            biases_init           =self.biases_init,
            streams_init          =self.streams_init,
            layers                =self.layers,
            weights_process_params=self.weights_process_params,
            weights_process_args  =self.weights_process_args,
        )

        # save to output path
        with open(os.path.join(self.output_path, f'src/{self.name}_top.cpp'),'w') as f:
            f.write(network_src)

        # set generated flag
        self.is_generated["source"] = True

    def generate_testbench(self):

        # format testbench code template
        network_tb_src = network_tb_src_template.format(
            name = self.name,
            NAME = self.name.upper(),
            input_data_path = f"{self.partition.layers[0].name}_in_0.dat",
            weights_reloading_path = f"{self.partition.weights_reloading_layer}_weights_0.dat",
            output_data_path = f"{self.partition.layers[-1].name}_out_0.dat"

        )

        # save to output path
        with open(os.path.join(self.output_path,f'tb/{self.name}_tb.cpp'),'w') as f:
            f.write(network_tb_src)

        # set generated flag
        self.is_generated["testbench"] = True

    def create_testbench_data(self, input_data):
        # get the input name and shape
        input_name  = self.sess.get_inputs()[0].name
        input_shape = self.sess.get_inputs()[0].shape
        # check data is right shape
        if input_shape != list(input_data.shape):
            raise ValueError(f"expected input shape: {input_shape}, data shape: {list(input_data.shape)}")
        # save input layer
        # TODO add bitwidth
        if len(self.partition.input_nodes) > 1:
            # check if multiple input nodes
            raise NotImplementedError("Multiple input nodes not currently supported.")
        input_node = self.partition.input_nodes[0]
        input_stream = np.array( self.sess.run([input_node], { input_name : input_data } )[0] )
        input_stream = np.moveaxis(input_stream, 1, -1)
        input_stream = onnx_data._convert_fixed_port_stream(input_stream.reshape(-1))
        print("Writing input stream to .dat file")
        onnx_data._fixed_point_stream_to_dat(input_stream,
                os.path.join(self.output_path, f"data/{self.partition.layers[0].name}_in"),
                streams=int(self.partition.layers[0].parameters.coarse_in), port_width=self.port_width)
        # save output layer
        if len(self.partition.output_nodes) > 1:
            # check if multiple output nodes
            raise NotImplementedError("Multiple output nodes not currently supported.")
        output_node = self.partition.output_nodes[0]
        output_raw = np.array( self.sess.run([output_node], { input_name : input_data } )[0] )
        print(f"[DEBUG] ONNX output node: {output_node}")
        print(f"[DEBUG] ONNX output raw shape (NCHW): {output_raw.shape}")
        output_nhwc = np.moveaxis(output_raw, 1, -1)  # (batch, rows, cols, channels)
        print(f"[DEBUG] ONNX output NHWC shape: {output_nhwc.shape}")
        wr_factor = self.partition.weights_reloading_factor
        total_channels = output_nhwc.shape[-1]
        channels_per_wr = total_channels // wr_factor
        print(f"[DEBUG] wr_factor={wr_factor}, total_channels={total_channels}, channels_per_wr={channels_per_wr}")
        output_spatial = output_nhwc.reshape(-1, total_channels)  # (batch*rows*cols, channels)
        # _transform_weights interleaves filters: WR pass j, position nf -> raw filter nf*wr_factor + j
        # So valid data for (pixel, wr_j, channel_nf) = ONNX output[pixel, nf*wr_factor + j]
        # Reshape to (N, channels_per_wr, wr_factor) then transpose to (N, wr_factor, channels_per_wr)
        output_pairs = output_spatial.reshape(output_spatial.shape[0], channels_per_wr, wr_factor)
        output_wr = output_pairs.transpose(0, 2, 1)    # (N, wr_factor, channels_per_wr)
        output_flat = output_wr.reshape(-1)             # N * wr_factor * channels_per_wr values
        print(f"[DEBUG] output_flat shape: {output_flat.shape} (expected {output_spatial.shape[0] * wr_factor * channels_per_wr})")
        print(f"[DEBUG] First 8 values per WR pass for pixel 0:")
        print(f"[DEBUG]   WR0 ch0-7: {output_wr[0,0,:8]}")
        if wr_factor > 1:
            print(f"[DEBUG]   WR1 ch0-7: {output_wr[0,1,:8]}")
        print(f"[DEBUG] Corresponding ONNX channels for pixel 0, WR0 ch0-7: {output_spatial[0, 0:16:2]}")
        if wr_factor > 1:
            print(f"[DEBUG] Corresponding ONNX channels for pixel 0, WR1 ch0-7: {output_spatial[0, 1:16:2]}")
        output_stream = onnx_data._convert_fixed_port_stream(output_flat)
        print("Writing valid output stream to .dat file")
        onnx_data._fixed_point_stream_to_dat(output_stream,
                os.path.join(self.output_path, f"data/{self.partition.layers[-1].name}_out"),
                streams=int(self.partition.layers[-1].parameters.coarse_out), port_width=self.port_width)

    """
    Vitis HLS functions
    """

    def create_vitis_hls_project(self, fpga_part="xc7z045ffg900-2", clk=5):

        # check everything is generated
        assert reduce(lambda a, b: a & b, self.is_generated.values()), "ERROR: not all stages are generated!"

        # pass parameters via environment variables (vitis-run does not support extra positional args)
        os.environ['HLS_PRJ_PATH']   = self.output_path
        os.environ['HLS_FPGA_PART']  = fpga_part
        os.environ['HLS_CLK_PERIOD'] = str(clk)

        # store for use by subsequent run methods
        self._fpga_part  = fpga_part
        self._clk_period = str(clk)

        # create hls project
        os.system(f"vitis-run --mode hls --tcl {self.fpgaconvnet_root}/scripts/hls/create_partition_project.tcl")

        # set project generated flag
        self.project_generated = True

    def _set_hls_env(self):
        os.environ['HLS_PRJ_PATH']   = self.output_path
        os.environ['HLS_FPGA_PART']  = self._fpga_part
        os.environ['HLS_CLK_PERIOD'] = self._clk_period

    def run_csynth(self):
        assert self.project_generated, "ERROR: project not yet created!"
        self._set_hls_env()
        os.system(f"vitis-run --mode hls --tcl {self.fpgaconvnet_root}/scripts/hls/run_csynth.tcl")

    def run_csim(self):
        assert self.project_generated, "ERROR: project not yet created!"
        self._set_hls_env()
        os.system(f"vitis-run --mode hls --tcl {self.fpgaconvnet_root}/scripts/hls/run_csim.tcl")

    def run_cosim(self):
        assert self.project_generated, "ERROR: project not yet created!"
        self._set_hls_env()
        os.system(f"vitis-run --mode hls --tcl {self.fpgaconvnet_root}/scripts/hls/run_cosim.tcl")

    def run_implementation(self):
        assert self.project_generated, "ERROR: project not yet created!"
        self._set_hls_env()
        os.system(f"vitis-run --mode hls --tcl {self.fpgaconvnet_root}/scripts/hls/run_implementation.tcl")

    def export_design(self):
        assert self.project_generated, "ERROR: project not yet created!"
        self._set_hls_env()
        os.system(f"vitis-run --mode hls --tcl {self.fpgaconvnet_root}/scripts/hls/export_design.tcl")

