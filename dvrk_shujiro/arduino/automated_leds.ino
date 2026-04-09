int NUM_PEG_LEDS = 8;

// 8 pegs, each with 2 LEDs (blue and white)
int pins[8][2] = {{25, 24}, {26, 27},  {31, 30}, {28, 29}, {45, 44}, {46, 47}, {49, 48}, {51, 50}}; // first pin is always blue, second pin is red

// 8 sensors on the target pegs (left/right), detecting when a cylinder is placed there
int inputPins[8] = {35, 34, 36, 37, 42, 43, 41, 40};

// 4 sensors on the center pegs, detecting when a cylinder is picked up from the starting position.
int inputPinsForCenter[4] = {23, 22, 52, 53};

// links each center peg to its two corresponding target pegs (left and right pair).
int indexPairings[4][2] = {{8, 1}, {7, 2}, {5, 3}, {6, 4}}; // pairs the center sleeves to the correct pins.

//  a pre-set sequence of 0s and 1s deciding whether the target color is blue or white for each trial
int whiteBlueAlternations[20] = {0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0};

// a pre-set list of 50 trials, each number (0–3) indicating which center peg is used.
int fifty_peg_trials[50] = {3, 3, 1, 3, 3, 2, 1, 0, 3, 1, 3, 2, 1, 3, 1, 1, 0, 0, 1, 0, 0, 3, 0, 1, 3, 0, 2, 1, 3, 1, 2, 1, 1, 3, 0, 3, 3, 2, 1, 0, 1, 1, 2, 3, 1, 3, 2, 0, 0, 3};

int currentColor = 0;
int currPin = 0;
bool loopRun = false;
unsigned long startTime;

void setup() {
  // put your setup code here, to run once:
  //
  pinMode(A0, INPUT);
  randomSeed(analogRead(millis()));
  for(int i = 0; i < NUM_PEG_LEDS; i++){
    for(int k = 0; k < 2; k++){
      pinMode(pins[i][k], OUTPUT);
      digitalWrite(pins[i][k], LOW);
    }
  }

  for(int k = 0; k <8; k++){
    pinMode(inputPins[k], INPUT);
  }

  for(int j = 0; j < 4; j++){
    pinMode(inputPinsForCenter[j], INPUT);
  }

  startTime = millis();
  Serial.begin(9600);
}

void resetLEDs(){
  for(int i = 0; i < 8; i++){
      digitalWrite(pins[i][0], LOW);
      digitalWrite(pins[i][1], LOW);
  }
}

void turnOnWhite(int pinNum){
  if(digitalRead(pins[pinNum - 1][1]) == 1){
    return;
  }
  digitalWrite(pins[pinNum - 1][1], HIGH);
}

void turnOnBlue(int pinNum){
  if(digitalRead(pins[pinNum - 1][0]) == 1){
    return;
  }
  digitalWrite(pins[pinNum - 1][0], HIGH);
}

void waitUntilPinOff(int centerPinIndex){
  while(digitalRead(inputPinsForCenter[centerPinIndex]) == 1){
    //Serial.print('+');
    //do nothing.
  }
  bool val = false;
  Serial.print("CENTER ");
  Serial.print(centerPinIndex + '0');
  Serial.print(",LIFTED,");
  Serial.print(millis() - startTime);
  Serial.print(",");
  Serial.println(val);
}

//specifically for change in any configuration, from high to low.
int waitUntilChangeCenter(){
  //record the initial configuration of each of the pins
  //note; this function should only exit if we go from a HIGH -> LOW state.
  int numHighPegs = 0;
  int initialConfig[4]; // initialConfig will only hold the pins that are normally high
  for(int i = 0; i < 4; i++){
    initialConfig[i] = digitalRead(inputPinsForCenter[i]);
    if(initialConfig[i] == 1){
      numHighPegs += 1;
    }
  }
  if(numHighPegs == 0){ return -1; }

  int pinChangedIndex = -1;
  bool stateChanged = false;
  while(!stateChanged){
    //iterate through the configuration, checking that they are all the same
    for(int k = 0; k < 4; k++){
      if(initialConfig[k] == 0){ continue; } //we only care about pegs that go from high to low.
      if(initialConfig[k] != digitalRead(inputPinsForCenter[k])){
        stateChanged = true;
        pinChangedIndex = k;
        break;
      }
    }
  }
  return pinChangedIndex;
}

void waitUntilPinOn(int inputPinNum){
  while (Serial.available() > 0) {
    Serial.read(); //flush pre-existing buffer
  }
  bool manualIntervention = false;
  while(digitalRead(inputPins[inputPinNum - 1]) == 0){
    if(Serial.available() > 0){
      while (Serial.available() > 0) {
        Serial.read();
      }
      manualIntervention = true;
      break;
    }
    //do nothing.
    //Serial.println("waiting...");
  }
  Serial.print("PEG ");
  Serial.print(inputPinNum + '0');
  Serial.print(",PLACED,");
  Serial.print(millis() - startTime);
  Serial.print(",");
  Serial.println(manualIntervention);
}

void waitUntilPinOnCenter(int inputPinIndex){
  while (Serial.available() > 0) {
    Serial.read(); //flush pre-existing buffer
  }
  bool manualIntervention = false;
  while(digitalRead(inputPinsForCenter[inputPinIndex]) == 0){
    if(Serial.available() > 0){
      while (Serial.available() > 0) {
        Serial.read();
      }
      manualIntervention = true;
      //then just break;
      break;
    }
    //do nothing.
    //Serial.println("waiting...");
  }
  Serial.print("CENTER ");
  Serial.print(inputPinIndex + '0');
  Serial.print(",PLACED,");
  Serial.print(millis() - startTime);
  Serial.print(",");
  Serial.println(manualIntervention);
}

int SEQUENCE_SIZE = 3; //@MK: change this to however long you want your sequence to be
int pinIndexSequence[3][2] = {{1, 0}, {2, 1}, {3, 0}}; //@MK; add sequences to this list. Also change the '3' in 'pinIndexSequence[3][2]' to whatever size you want the sequence
//each array is {pinIndex, color}, where color is 0 if blue, and 1 if white.


void startSequence(){
  //input tells us which pin to look out for!
  for(int i = 0; i < SEQUENCE_SIZE; i++){
    int pinNum = pinIndexSequence[i][0];
    if(pinIndexSequence[i][1] == 0){
      resetLEDs();
      turnOnBlue(pinNum);
    } else {
      resetLEDs();
      turnOnWhite(pinNum);
    }
    waitUntilPinOn(pinNum);
  }
}


void runOneTrialSamePeg(int centerPinIndex){
  turnOnBlue(indexPairings[centerPinIndex][0]);
  turnOnBlue(indexPairings[centerPinIndex][1]); //turn them both on!
  waitUntilPinOff(centerPinIndex);

  resetLEDs();

  int pinToLight = indexPairings[centerPinIndex][random(0, 2)]; // the pin index to light up

  //then randmly decide which color to have it on!
  if(whiteBlueAlternations[currentColor] == 0){
    turnOnBlue(pinToLight);
  } else {
    turnOnWhite(pinToLight);
  }
  currentColor += 1;
  if(currentColor == 20){
    currentColor = 0;
  }

  waitUntilPinOn(pinToLight);
  resetLEDs();
  waitUntilPinOnCenter(centerPinIndex);
  turnOnWhite(indexPairings[centerPinIndex][0]);
  turnOnWhite(indexPairings[centerPinIndex][1]); //turn them both on!
  delay(2000);
  resetLEDs();
}

void runSequences(){
  int pinChangedIndex = waitUntilChangeCenter();
    //Serial.println("Game piece lifted from start!");
    if(pinChangedIndex != -1){
      startSequence();
    }
    while(digitalRead(inputPinsForCenter[pinChangedIndex]) == 0){
      //wait till the pin returns back to start
    }
    Serial.println("Reset the LEDS!");
    resetLEDs();
}


void loop() {
  // put your main code here, to run repeatedly:
  if(!loopRun){
    int i = 0;
    while(i < 50){
      runOneTrialSamePeg(fifty_peg_trials[i]);
      i+=1;
    }
  }
  loopRun = true; 
}
