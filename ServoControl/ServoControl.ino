#include <Servo.h>
#include "lookupTable.h" // Include the generated header file

struct LookupTableEntry {
    float theta;
    float si;
    float phi;
};

extern const LookupTableEntry lookupTable[]; // Declare the lookup table

Servo myServo;

void setup() {
    myServo.attach(9);  // Attach servo to pin 9
}

void loop() {
    float theta = /* get the value of theta from some source */;
    float si = /* get the value of si from some source */;

    float phi = interpolatePhi(theta, si);
    myServo.write(phi);

    delay(1000);
}

float interpolatePhi(float theta, float si) {
    float phiTheta = interpolateForTheta(theta);
    float phiSi = interpolateForSi(si);

    return (phiTheta + phiSi) / 2; // Averaging the two phi values
}

float interpolateForTheta(float theta) {
    theta = fmod(theta, 360.0); // Normalize theta to [0, 360)
    if (theta < 0) theta += 360.0;

    float nearestLower = -1;
    float nearestHigher = -1;
    float phiLower = 0;
    float phiHigher = 0;

    for (int i = 0; i < (sizeof(lookupTable) / sizeof(lookupTable[0])); i++) {
        float tableTheta = fmod(lookupTable[i].theta, 360.0);
        if (tableTheta < 0) tableTheta += 360.0;

        if (tableTheta <= theta) {
            if (nearestLower == -1 || tableTheta > nearestLower) {
                nearestLower = tableTheta;
                phiLower = lookupTable[i].phi;
            }
        }

        if (tableTheta >= theta) {
            if (nearestHigher == -1 || tableTheta < nearestHigher) {
                nearestHigher = tableTheta;
                phiHigher = lookupTable[i].phi;
            }
        }
    }

    // Handle wrap-around for theta
    if (nearestLower == -1) nearestLower = 360.0;
    if (nearestHigher == -1) nearestHigher = 0.0;

    // Linear interpolation
    return phiLower + (phiHigher - phiLower) * (theta - nearestLower) / (nearestHigher - nearestLower);
}

float interpolateForSi(float si) {
    float nearestLower = -1;
    float nearestHigher = -1;
    float phiLower = 0;
    float phiHigher = 0;

    for (int i = 0; i < (sizeof(lookupTable) / sizeof(lookupTable[0])); i++) {
        float tableSi = lookupTable[i].si;

        if (tableSi <= si) {
            if (nearestLower == -1 || tableSi > nearestLower) {
                nearestLower = tableSi;
                phiLower = lookupTable[i].phi;
            }
        }

        if (tableSi >= si) {
            if (nearestHigher == -1 || tableSi < nearestHigher) {
                nearestHigher = tableSi;
                phiHigher = lookupTable[i].phi;
            }
        }
    }

    // Linear interpolation
    if (nearestLower != -1 && nearestHigher != -1) {
        return phiLower + (phiHigher - phiLower) * (si - nearestLower) / (nearestHigher - nearestLower);
    } else {
        // Return one of the values if only one bound is found
        return nearestLower != -1 ? phiLower : phiHigher;
    }
}
