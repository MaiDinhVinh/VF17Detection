#include <Servo.h>

#define SERVO_PIN 9

Servo myservo;

const int HOME_ANGLE = 0;
const int PUSH_ANGLE = 180;

void setup() {
  Serial.begin(9600);

  myservo.attach(SERVO_PIN); //tin hieu
  myservo.write(HOME_ANGLE);
}

void loop() {

  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == '1') {
      myservo.write(HOME_ANGLE);
      delay(1000);             
      myservo.write(PUSH_ANGLE);
    }
  }
}
