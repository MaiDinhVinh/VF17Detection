#include <Servo.h>

/* ---------------- Pins ---------------- */
const uint8_t PIN_SERVO = 9;   // Servo signal pin
const uint8_t PIN_TRIG  = 6;   // Ultrasonic TRIG
const uint8_t PIN_ECHO  = 7;   // Ultrasonic ECHO

/* -------- Servo motion (tune) --------- */
const int REST_ANGLE      = 20;   // retracted
const int PUSH_ANGLE      = 120;  // fully extended
const uint16_t PUSH_MS    = 280;  // hold extended
const uint16_t RETRACT_MS = 250;  // cooldown

/* -------- Ultrasonic (tune) ----------- */
const float DETECT_CM          = 12.0; // detection threshold
const uint8_t DEBOUNCE_COUNT   = 3;    // consecutive hits needed
const uint16_t MEASURE_PERIOD  = 30;   // ms between reads

/* -------------- Serial ---------------- */
const long BAUD = 9600;

/* -------------- Queue ----------------- */
const uint8_t Q_CAP = 32;
uint8_t qBuf[Q_CAP]; // 0 = rotten, 1 = fresh
uint8_t qHead = 0, qTail = 0, qSize = 0;

bool enqueue(uint8_t v) {
  if (qSize >= Q_CAP) return false;
  qBuf[qTail] = v;
  qTail = (qTail + 1) % Q_CAP;
  qSize++;
  return true;
}

bool dequeue(uint8_t &out) {
  if (qSize == 0) return false;
  out = qBuf[qHead];
  qHead = (qHead + 1) % Q_CAP;
  qSize--;
  return true;
}

/* ---------- Integer state IDs ---------- */
const int STATE_IDLE       = 0;
const int STATE_PUSHING    = 1;
const int STATE_RETRACTING = 2;
const int STATE_COOLDOWN   = 3;

int state = STATE_IDLE;
unsigned long stateTs = 0;

/* ----------- Other globals ------------ */
Servo Nigga;
unsigned long lastMeasure = 0;
uint8_t belowCount = 0;
bool objectPresent = false;

/* ------------- Functions -------------- */
float readDistanceCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  unsigned long dur = pulseIn(PIN_ECHO, HIGH, 30000UL); // 30 ms timeout
  if (dur == 0) return 9999.0;
  return (dur * 0.0343f) * 0.5f; // convert to cm
}

void setState(int s) {
  state = s;
  stateTs = millis();
}

void handleSerial() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c == '0' || c == '1') {
      uint8_t v = (c == '0') ? 0 : 1;
      if (!enqueue(v)) {
        Serial.println(F("QUEUE_FULL"));
      } else {
        Serial.print(F("ENQ:"));
        Serial.println(v == 0 ? F("ROTTEN") : F("FRESH"));
      }
    }
  }
}

/* ------------- Setup ------------------ */
void setup() {
  Serial.begin(BAUD);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_LED, OUTPUT);

  Nigga.attach(PIN_SERVO);
  Nigga.write(REST_ANGLE);
  digitalWrite(PIN_LED, LOW);

  delay(500);
  Serial.println(F("READY"));
}

/* ------------- Loop ------------------- */
void loop() {
  handleSerial();

  unsigned long now = millis();
  if (now - lastMeasure >= MEASURE_PERIOD) {
    lastMeasure = now;
    float d = readDistanceCm();

    if (d < DETECT_CM) {
      if (belowCount < 255) belowCount++;
    } else {
      if (belowCount > 0) belowCount--;
    }

    bool presentNow = (belowCount >= DEBOUNCE_COUNT);

    // Fruit just arrived
    if (presentNow && !objectPresent && state == STATE_IDLE) {
      uint8_t lbl = 1; // default fresh
      if (!dequeue(lbl)) {
        Serial.println(F("QUEUE_EMPTY_DEFAULT_FRESH"));
        lbl = 1;
      }

      Serial.print(F("OBJ:"));
      Serial.println(lbl == 0 ? F("ROTTEN") : F("FRESH"));

      if (lbl == 0) {
        // rotten -> push
        Nigga.write(PUSH_ANGLE);
        digitalWrite(PIN_LED, HIGH);
        setState(STATE_PUSHING);
      }
      // fresh -> do nothing (still dequeued)
    }
    objectPresent = presentNow;
  }

  // State machine
  switch (state) {
    case STATE_IDLE:
      break;

    case STATE_PUSHING:
      if (millis() - stateTs >= PUSH_MS) {
        Nigga.write(REST_ANGLE);
        setState(STATE_RETRACTING);
      }
      break;

    case STATE_RETRACTING:
      if (millis() - stateTs >= RETRACT_MS) {
        setState(STATE_COOLDOWN);
      }
      break;

    case STATE_COOLDOWN:
      if (!objectPresent) {
        digitalWrite(PIN_LED, LOW);
        setState(STATE_IDLE);
      }
      break;
  }
}
