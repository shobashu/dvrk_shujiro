int NUM_PEG_LEDS = 8;
int pins[8][2] = {{51, 50}, {48, 49}, {46, 47}, {45, 44}, {24, 25}, {26, 27}, {28, 29}, {30, 31}}; // [0]=blue, [1]=white
int inputPins[8] = {41, 42, 43, 40, 32, 33, 23, 22};
int inputPinsForCenter[4] = {39, 38, 52, 53};

// ── Single-cylinder, fixed-peg mode ─────────────────────────────────────────
// Only ONE center peg and ONE outer target peg are ever waited on — every
// other peg on the board is inert. The cylinder always travels the same
// center-peg-3 <-> outer-peg-3 path; only the cue COLOR varies per trial.
// This is what makes state detection (CylinderStateTracker, single
// largest-box detection) and the depth-camera pipeline (single --reuse-
// prompts template, see 09_live_trial_monitor.py) both unambiguous.
//
// ACTIVE_TARGET_PEG = 2 (physical outer peg 3) keeps the same numeric index
// as ACTIVE_CENTER_PEG, matching PEG_CLASSIFICATIONS in config.py
// ({1:"L",2:"L",3:"L",4:"L",...}) so the orientation-check's left-camera
// crop keeps working with zero Python-side changes.
const int ACTIVE_CENTER_PEG = 2;   // index into inputPinsForCenter -> physical Center Peg 3
const int ACTIVE_TARGET_PEG = 2;   // index into inputPins/pins     -> physical Outer Peg 3

const int NUM_TRIALS = 30;
int currentTrial = 0;
bool experimentRunning = false;

// Only the cue color varies per trial now (0=blue, 1=white) — target peg is
// fixed to ACTIVE_TARGET_PEG. Edit freely, this is just a reasonable
// starting sequence.
int trials[NUM_TRIALS] = {
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1
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

  Serial.println("--- System Ready (single-cylinder, fixed-peg mode) ---");
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

// Only checks the ONE active center peg and the ONE active target peg — the
// rest of the board is out of play and deliberately ignored.
bool checkBoardState() {
  bool isReady = true;
  if (digitalRead(inputPinsForCenter[ACTIVE_CENTER_PEG]) == 1) {
    Serial.print("ERROR: Center Peg "); Serial.print(ACTIVE_CENTER_PEG + 1); Serial.println(" is empty.");
    isReady = false;
  }
  if (digitalRead(inputPins[ACTIVE_TARGET_PEG]) == 0) {
    Serial.print("ERROR: Outer Peg "); Serial.print(ACTIVE_TARGET_PEG + 1); Serial.println(" has an object on it.");
    isReady = false;
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

  int targetIdx   = ACTIVE_TARGET_PEG;
  int targetColor = trials[currentTrial];
  // Decoy = the mirrored peg on the OTHER (inactive) row, same as the
  // original 8-peg board's oppositeIdx pairing. It's lit purely as a visual
  // decoy — its sensor is never checked/waited on, you never actually place
  // anything there, it just turns off again the moment you lift.
  int decoyIdx = (targetIdx + 4) % 8;

  unsigned long cueTime = 0;
  unsigned long liftTime = 0;
  unsigned long placeTime = 0;

  Serial.print("\n--- Trial "); Serial.print(currentTrial + 1); Serial.println(" ---");
  Serial.print("Waiting for lift from Center Peg "); Serial.println(ACTIVE_CENTER_PEG + 1);
  Serial.print("CUE,"); Serial.println(targetColor == 0 ? "Blue" : "White");

  // 1. Indicate the pair — target + decoy, both same color. The color alone
  //    doesn't tell you which is real yet; that's only resolved once you
  //    lift and the decoy turns off.
  cueTime = millis();
  digitalWrite(pins[targetIdx][targetColor], HIGH);
  digitalWrite(pins[decoyIdx][targetColor], HIGH);

  // 2. Wait for Lift from Center & Record Lift Time
  if (!waitForPin(inputPinsForCenter[ACTIVE_CENTER_PEG], 1)) { abortExperiment(); return; }
  liftTime = millis();
  Serial.println("LIFTED");

  Serial.print("Object lifted. Move to Outer Peg "); Serial.println(targetIdx + 1);

  // 3. Decoy off, only the real target stays lit.
  resetLEDs();
  digitalWrite(pins[targetIdx][targetColor], HIGH);

  // 4. Wait for Place on Target Outer & Record Place Time
  if (!waitForPin(inputPins[targetIdx], 0)) { abortExperiment(); return; }
  placeTime = millis();

  Serial.println("Target hit. Please pick the object back up.");
  digitalWrite(pins[targetIdx][targetColor], LOW);

  // 5. Wait for Lift from Target Outer
  if (!waitForPin(inputPins[targetIdx], 1)) { abortExperiment(); return; }

  Serial.print("Object picked up. Waiting for return to Center Peg "); Serial.println(ACTIVE_CENTER_PEG + 1);

  // 6. Wait for Return to Center — the cylinder MUST go back to the same
  //    center peg before the trial is logged and the next one starts.
  if (!waitForPin(inputPinsForCenter[ACTIVE_CENTER_PEG], 0)) { abortExperiment(); return; }

  Serial.println("Object returned successfully.");

  // --- OUTPUT CSV DATA --- (same schema the Python side already parses)
  Serial.print("DATA,");
  Serial.print(currentTrial + 1); Serial.print(",");
  Serial.print(targetIdx + 1); Serial.print(",");
  Serial.print(targetColor == 0 ? "Blue" : "White"); Serial.print(",");
  Serial.print(cueTime); Serial.print(",");
  Serial.print(liftTime); Serial.print(",");
  Serial.println(placeTime);

  Serial.println("2 second pause...");

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
