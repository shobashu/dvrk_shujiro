int NUM_PEG_LEDS = 8;
int pins[8][2] = {{51, 50}, {48, 49}, {46, 47}, {45, 44}, {24, 25}, {26, 27}, {28, 29}, {30, 31}}; // [0]=blue, [1]=white
int inputPins[8] = {41, 42, 43, 40, 32, 33, 23, 22};
int inputPinsForCenter[4] = {39, 38, 52, 53};

const int NUM_TRIALS = 30;
int currentTrial = 0;
bool experimentRunning = false;

// Trial Format: {centerPegIndex (0-3), targetOuterPegIndex (0-7), targetColor (0=blue, 1=white)}
int trials[NUM_TRIALS][3] = {
  {1, 5, 1}, {0, 4, 0}, {2, 2, 1}, {3, 7, 0}, {0, 0, 1}, 
  {1, 1, 0}, {2, 6, 1}, {3, 3, 0}, {0, 4, 1}, {1, 5, 0}, 
  {2, 2, 0}, {3, 7, 1}, {0, 0, 0}, {1, 1, 1}, {2, 6, 0}, 
  {3, 3, 1}, {1, 5, 1}, {0, 4, 0}, {2, 2, 1}, {3, 7, 0},
  {0, 0, 1}, {1, 1, 0}, {2, 6, 1}, {3, 3, 0}, {0, 4, 1},
  {1, 5, 0}, {2, 2, 0}, {3, 7, 1}, {0, 0, 0}, {1, 1, 1}
};

void setup() {
  Serial.begin(9600);
  
  for(int i = 0; i < NUM_PEG_LEDS; i++){
    pinMode(pins[i][0], OUTPUT);
    pinMode(pins[i][1], OUTPUT);
    digitalWrite(pins[i][0], LOW);
    digitalWrite(pins[i][1], LOW);
  }
  
  for(int k = 0; k < 8; k++){
    pinMode(inputPins[k], INPUT_PULLUP);
  }
  
  for(int j = 0; j < 4; j++){
    pinMode(inputPinsForCenter[j], INPUT_PULLUP);
  }

  Serial.println("--- System Ready ---");
  Serial.println("Type 's' to START.");
  Serial.println("Type 'q' at any time to ABORT.");
}

void resetLEDs(){
  for(int i = 0; i < 8; i++){
      digitalWrite(pins[i][0], LOW);
      digitalWrite(pins[i][1], LOW);
  }
}

void abortExperiment() {
  Serial.println("\n*** EXPERIMENT ABORTED BY USER ***");
  Serial.println("Type 's' to start a new block from Trial 1.");
  resetLEDs();
  experimentRunning = false;
}

bool waitForPin(int pin, int targetState) {
  while(digitalRead(pin) != targetState) {
    if (Serial.available() > 0) {
      char cmd = Serial.read();
      if (cmd == 'q' || cmd == 'Q') {
        return false;
      }
    }
    delay(10);
  }
  return true; 
}

bool checkBoardState() {
  bool isReady = true;
  for(int i = 0; i < 4; i++){
    if(digitalRead(inputPinsForCenter[i]) == 1) {
      Serial.print("ERROR: Center Peg "); Serial.print(i + 1); Serial.println(" is empty.");
      isReady = false;
    }
  }
  for(int k = 0; k < 8; k++){
    if(digitalRead(inputPins[k]) == 0) {
      Serial.print("ERROR: Outer Peg "); Serial.print(k + 1); Serial.println(" has an object on it.");
      isReady = false;
    }
  }
  return isReady;
}

void runTrial() {
  if (currentTrial >= NUM_TRIALS) {
    Serial.println("\n*** All 30 trials completed! ***");
    Serial.println("Type 's' to run a new 30-trial block.");
    experimentRunning = false;
    return;
  }

  int centerIdx = trials[currentTrial][0];
  int targetIdx = trials[currentTrial][1];
  int targetColor = trials[currentTrial][2];
  int oppositeIdx = (targetIdx + 4) % 8; 

  unsigned long cueTime = 0;
  unsigned long liftTime = 0;
  unsigned long placeTime = 0;

  Serial.print("\n--- Trial "); Serial.print(currentTrial + 1); Serial.println(" ---");
  Serial.print("Waiting for lift from Center Peg "); Serial.println(centerIdx + 1);

  // 1. Indicate Pair & Record Cue Time
  cueTime = millis();
  digitalWrite(pins[targetIdx][targetColor], HIGH);
  digitalWrite(pins[oppositeIdx][targetColor], HIGH);

  // 2. Wait for Lift from Center & Record Lift Time
  if (!waitForPin(inputPinsForCenter[centerIdx], 1)) { abortExperiment(); return; }
  liftTime = millis();
  Serial.println("LIFTED");   // add this line
  
  Serial.print("Object lifted. Move to Outer Peg "); Serial.println(targetIdx + 1);

  // 3. Indicate Specific Target
  resetLEDs(); 
  digitalWrite(pins[targetIdx][targetColor], HIGH); 

  // 4. Wait for Place on Target Outer & Record Place Time
  if (!waitForPin(inputPins[targetIdx], 0)) { abortExperiment(); return; }
  placeTime = millis();

  Serial.println("Target hit. Please pick the object back up.");
  digitalWrite(pins[targetIdx][targetColor], LOW); 

  // 5. Wait for Lift from Target Outer
  if (!waitForPin(inputPins[targetIdx], 1)) { abortExperiment(); return; }

  Serial.print("Object picked up. Waiting for return to Center Peg "); Serial.println(centerIdx + 1);

  // 6. Wait for Return to Center
  if (!waitForPin(inputPinsForCenter[centerIdx], 0)) { abortExperiment(); return; }

  Serial.println("Object returned successfully.");
  
  // --- OUTPUT CSV DATA ---
  // Format: DATA, TrialNum, TargetPeg, TargetColor, CueMs, LiftMs, PlaceMs
  Serial.print("DATA,");
  Serial.print(currentTrial + 1); Serial.print(",");
  Serial.print(targetIdx + 1); Serial.print(",");
  Serial.print(targetColor == 0 ? "Blue" : "White"); Serial.print(",");
  Serial.print(cueTime); Serial.print(",");
  Serial.print(liftTime); Serial.print(",");
  Serial.println(placeTime);

  Serial.println("2 second pause...");
  
  // 7. Pause between trials
  for(int d = 0; d < 200; d++) {
    if (Serial.available() > 0) {
      char cmd = Serial.read();
      if (cmd == 'q' || cmd == 'Q') {
        abortExperiment(); return; 
      }
    }
    delay(10);
  }

  currentTrial++; 
}

void loop() {
  if (!experimentRunning) {
    if (Serial.available() > 0) {
      char cmd = Serial.read();
      if (cmd == 's' || cmd == 'S') {
        Serial.println("\nChecking board state...");
        if (checkBoardState()) {
          Serial.println("Board state OK! Starting experiment in 2 seconds...");
          
          // Send SYNC pulse so Python can calculate true Unix time
          Serial.print("SYNC,"); 
          Serial.println(millis());
          
          delay(2000);
          experimentRunning = true;
          currentTrial = 0; 
        } else {
          Serial.println("Board state check failed. Please fix the pieces and type 's' again.");
        }
      }
    }
  } else {
    runTrial();
  }
}