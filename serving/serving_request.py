
# output_path = "/home/crr/workspace/sik_models/outputs/to_deploy"
# model_name = "BN_Inception_sig"
# version = "1"
# save_path = os.path.join(output_path, model_name, version)
# tf.saved_model.save(model, save_path, signatures=serving)

#%% Docker
# docker run -v /home/crr/workspace/sik_models/outputs/tfx_model:/models/tfx_model -p 8500:8500 -p 8501:8501 -e MODEL_NAME=tfx_model -t -d --rm tensorflow/serving

# client = docker.from_env()
# client.containers.run("tensorflow/serving",
#                       volumes={
#                           output_path: {
#                               "bind": "/models",
#                               "mode": "rw"}},
#                       ports={"8501/tcp": 8501, "8500/tcp": 8500},
#                       environment={"MODEL_NAME": model_name},
#                       name=model_name,
#                       tty=True,
#                       detach=True,
#                       remove=True,
#                       stderr=True
#                       )

# %% Request metadata
# response = requests.get("http://localhost:8501/v1/models/tfx_model_sig/metadata")
# print(response.text)

# %% Read images
# imgpaths = ["/home/crr/datasets/Dataset/prod (5)/Ref_pic/46.jpg", "/home/crr/datasets/Dataset/prod (5)/Match/0.jpg"]
# bytes_input_images = [tf.io.read_file(imgpath).numpy() for imgpath in imgpaths]
#
#
# # %% REST API
# b64_input_images = [tf.io.encode_base64(input_image) for input_image in bytes_input_images]
# b64_utf_input_images = [input_image.numpy().decode("utf-8") for input_image in b64_input_images]
#
#
# data = json.dumps({"signature_name": "serving_default", "instances": b64_utf_input_images})
# headers = {"content-type": "application/json"}
# response = requests.post("http://localhost:8501/v1/models/tfx_model_sig:predict", data=data, headers=headers)
# scores = np.array(response.json()["predictions"])
#
# # %% gRPC API
# server = "localhost:8500"
# channel = grpc.insecure_channel(server)
# stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
#
# request = predict_pb2.PredictRequest()
# request.model_spec.name = "tfx_model_sig"
# request.model_spec.signature_name = "serving_default"
# img_proto = tf.make_tensor_proto(bytes_input_images, shape=np.shape(bytes_input_images))
# request.inputs["base64_image_bytes"].CopyFrom(img_proto)
#
# result = stub.Predict(request, 10.0)
# outputs = result.outputs["output_0"]
# shape = [d.size for d in outputs.tensor_shape.dim]
# scores = np.array(outputs.float_val).reshape(shape)
