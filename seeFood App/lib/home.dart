import 'package:flutter/material.dart';
import "dart:io";
import "package:image_picker/image_picker.dart";
import "package:tflite/tflite.dart";

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {

  late File _image;
  final picker = ImagePicker();
  bool _loading = false;
  List _output = [];

  pickImage() async {
    var image = await picker.getImage(source: ImageSource.camera);

    if (image == null) return null;
    setState(() {
      _image = File(image.path);
    });
    classifyImage(_image);
    print("PATH");
    print(_image.path);
  }

  pickGalleryImage() async {
    var image = await picker.getImage(source: ImageSource.gallery);

    if (image == null) return null;
    setState(() {
      _image = File(image.path);
    });

    classifyImage(_image);
    print("PATH");
    print(_image.path);
  }


  loadModel() async {
    await Tflite.loadModel(
        model: "assets/model_unquant.tflite", labels: "assets/labels.txt");
  }

  @override
  void initState() {
    super.initState();
    _loading = true;
    loadModel().then((value) {

    });
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  classifyImage(File image) async {
    var output = await Tflite.runModelOnImage(path: image.path,
        numResults: 2,
        threshold: 0.5,
        imageMean: 127.5,
        imageStd: 127.5);
    setState(() {
      _loading = false;
      _output = output!;
    });
  }


  @override
  Widget build(BuildContext context) {

    return Scaffold(
      backgroundColor: Color(0xFF101010),
      body: Container(
        padding: EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
          SizedBox(
          height: 100,
        ),
        _loading ? Center(child: Container(height: 80, child: Text("SeeFood",style: TextStyle(color: Colors.orangeAccent,fontSize: 42,),),)) : Stack(
          children: [
            Container(
              width: double.infinity,
              height: 80,
              decoration: _output[0]["label"] == "HotDog" ? BoxDecoration(color: Colors.green) : BoxDecoration(color: Colors.red) ,
              child: _output[0]["label"] == "HotDog" ? Center(child: Text("Hot-Dog", style: TextStyle(color: Colors.white, fontSize: 38,fontWeight: FontWeight.bold),))
              : Center(child: Text("Not Hot-Dog", style: TextStyle(color: Colors.yellow, fontSize: 38,fontWeight: FontWeight.bold),))
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(130, 60, 8, 0),
              child: ClipOval(
                  child: Container(
                      color: Colors.white,
                      child:  _output[0]["label"] == "HotDog" ? Image.asset(
                        "assets/check.png",
                        fit: BoxFit.cover,
                        width: 80.0,
                        height: 80.0,
                      ) : Image.asset(
                        "assets/cancel.png",
                        fit: BoxFit.cover,
                        width: 80.0,
                        height: 80.0,
                      )
                  )
              ),
            )
          ],
        ),
        SizedBox(height: 50,),
        Center(child:
        Container(
          width: 300,
          child: Column(
            children: [
              _loading ? Image.asset("assets/SiliconValley.jpg",) :
              Column(children: [
                Container(height: 300,
                  child: Image.file(_image),
                ),
              ]),
              SizedBox(height: 50,),

              Container(
                width: MediaQuery
                    .of(context)
                    .size
                    .width,
                child: Column(children: [
                  GestureDetector(
                    onTap: pickImage,
                    child: Container(
                      width: MediaQuery
                          .of(context)
                          .size
                          .width - 260,
                      alignment: Alignment.center,
                      padding: EdgeInsets.symmetric(
                          horizontal: 24, vertical: 17),
                      decoration: BoxDecoration(color: Color(0xFFE99600),
                          borderRadius: BorderRadius.circular(6)),
                      child: Text("Take a photo", style: TextStyle(
                          color: Colors.white),),
                    ),
                  ),
                  SizedBox(height: 10,),
                  GestureDetector(
                    onTap: pickGalleryImage,
                    child: Container(
                      width: MediaQuery
                          .of(context)
                          .size
                          .width - 260,
                      alignment: Alignment.center,
                      padding: EdgeInsets.symmetric(
                          horizontal: 24, vertical: 17),
                      decoration: BoxDecoration(color: Color(0xFFE99600),
                          borderRadius: BorderRadius.circular(6)),
                      child: Text("Camera", style: TextStyle(
                          color: Colors.white),),
                    ),
                  ),

                ],),
              )
            ],)

        ),
        ),
      ],
    ),)
    ,
    );
  }
}
