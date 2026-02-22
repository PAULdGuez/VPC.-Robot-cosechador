#include <ESP32Servo.h>

// Definición de pines
#define SERVO1_PIN 13
#define SERVO2_PIN 12
#define SERVO3_PIN 14
#define BAUDRATE 115200
#define SERVO_SPEED 10  // ms por grado (ajustar según tus servos)

Servo servo1, servo2, servo3;
TaskHandle_t Task1;

void setup() {
  Serial.begin(BAUDRATE);
  
  // Configuración de servos
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  servo3.setPeriodHertz(50);
  
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);
  servo3.attach(SERVO3_PIN, 500, 2400);

  // Crear tarea para el segundo núcleo
  xTaskCreatePinnedToCore(
    servo2Task,
    "Servo2Task",
    2048,
    NULL,
    1,
    &Task1,
    1
  );

  Serial.println("\nEnvie angulos en formato: theta1,theta2,theta3");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    int commaPos = input.indexOf(',');
    int comaPosDos = input.indexOf(',', commaPos + 1);

    if (commaPos > 0 && comaPosDos > 0) {
      int angulo1 = input.substring(0, commaPos).toInt();
      int angulo2 = input.substring(commaPos + 1, comaPosDos).toInt();
      int angulo3 = input.substring(comaPosDos + 1).toInt();
      
      if (angulo1 >= 0 && angulo1 <= 180 && 
          angulo2 >= 20 && angulo2 <= 180 && 
          angulo3 >= 0 && angulo3 <= 180) {
        
        // 1. Mover servo1 primero
        servo1.write(angulo1);
        
        // 2. Mover servo2 y servo3 simultánmente
        servo3.write(angulo3);
        xTaskNotify(Task1, angulo2, eSetValueWithOverwrite);
        
      } else {
        Serial.println("Los angulos deben estar entre 0-180°");
      }
    } else {
      Serial.println("Formato incorrecto. Usar: angulo1,angulo2,angulo3");
    }
  }
}

void servo2Task(void *pvParameters) {
  while (1) {
    uint32_t angulo;
    if (xTaskNotifyWait(0, ULONG_MAX, &angulo, portMAX_DELAY) == pdTRUE) {
      servo2.write(angulo);
    }
  }
}