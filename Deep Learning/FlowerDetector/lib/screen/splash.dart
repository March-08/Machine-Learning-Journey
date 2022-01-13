import 'package:flower_detector/screen/home.dart';
import 'package:flutter/material.dart';
import "package:splashscreen/splashscreen.dart";

class MySplash extends StatefulWidget {
  const MySplash({Key? key}) : super(key: key);

  @override
  _MySplashState createState() => _MySplashState();
}

class _MySplashState extends State<MySplash> {
  @override
  Widget build(BuildContext context) {
    return SplashScreen(
      gradientBackground: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          stops: [0.004, 1],
          colors: [Color(0xFFa8e063), Color(0xFF56ab2f)]),
      seconds: 3,
      title: Text(
        "Flower Detector",
        style: TextStyle(
            fontWeight: FontWeight.bold, fontSize: 30, color: Colors.white),
      ),
      image: Image.asset("assets/flower.png"),
      loaderColor: Colors.white,
      navigateAfterSeconds: Home(),
    );
  }
}
