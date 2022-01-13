import 'package:flower_detector/screen/splash.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flower Recognizer',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home:  MySplash(),
    );
  }
}
