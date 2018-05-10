#include "glew.h"
#include "freeglut.h"
#include "glm.hpp"
#include "ext.hpp"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>

#include "Shader_Loader.h"
#include "Render_Utils.h"
#include "Camera.h"
#include "Texture.h"

using namespace std;
using namespace glm;

GLuint programColor;
GLuint programTexture;

Core::Shader_Loader shaderLoader;

obj::Model shipModel;
obj::Model sphereModel;

float cameraAngle = 0;
glm::vec3 cameraPos = glm::vec3(-5, 0, 0);
glm::vec3 cameraDir;

glm::mat4 cameraMatrix, perspectiveMatrix;

glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, -0.9f, -1.0f));

int t = 0;
int frame = 0;

struct Particle
{
	glm::vec3 pos;
	glm::vec3 vel;
	glm::vec3 rot;
};


struct RigidBody {
	// force / torque function format
	typedef vec3(*Function) (
		double, // time of application
		vec3, // position
		quat, // orientation
		... // whole state of body, one var at a time
		);

	RigidBody(double m, mat3 inertia, Function force, Function torque);
	
	// Runge - Kutta fourth order differential equation solver
	void Update(double t, double dt);

	protected:
		// convert (Q,P,L) to (R,V,W)
		void Convert(quat Q, vec3 P, vec3 L, mat3 & R, vec3 & V, vec3 & W) const;
		// constant quantities
		double m_mass, m_invMass;
		mat3 m_inertia, m_invInertia;
		// state variables
		vec3 m_X; // position
		quat m_Q; // orientation
		vec3 m_P; // linear momentum
		vec3 m_L; // angular momentum
		
		// derived state variables
		mat3 m_R; // orientation matrix
		vec3 m_V; // linear velocity vector
		vec3 m_W; // angular velocity
	
		// force and torque functions
		Function m_force; Function m_torque;
};


void RigidBody::Convert(quat Q, vec3 P, vec3 L, mat3 & R, vec3 & V, vec3 & W) const {
	R = mat3_cast(Q);
	V = m_invMass *P;
	W = R* m_invInertia * transpose(R)*L;
}
void RigidBody::Update(double t, double dt) {
	double halfdt = 0.5 * dt, sixthdt = dt / 6.0;
	double tphalfdt = t + halfdt, tpdt = t + dt;

	vec3 XN, PN, LN, VN, WN;
	quat QN;
	mat3 RN;
	// A1 = G(t,S0), B1 = S0 + (dt / 2) * A1
	vec3 A1DXDT = m_V;
	quat A1DQDT = 0.5 * m_W * m_Q;
	vec3 A1DPDT = m_force(t, m_X, m_Q, m_P, m_L, m_R, m_V, m_W);
	vec3 A1DLDT = m_torque(t, m_X, m_Q, m_P, m_L, m_R, m_V, m_W);
	XN = m_X + halfdt * A1DXDT;
	QN.x = m_Q.x + halfdt * A1DQDT.x;
	QN.y = m_Q.y + halfdt * A1DQDT.y;
	QN.z = m_Q.z + halfdt * A1DQDT.z;
	QN.w = m_Q.w + halfdt * A1DQDT.w;

	PN = m_P + halfdt * A1DPDT;
	LN = m_L + halfdt * A1DLDT;
	Convert(QN, PN, LN, RN, VN, WN);

	// A2 = G(t + dt / 2,B1), B2 = S0 + (dt / 2) * A2
	vec3 A2DXDT = VN;
	quat A2DQDT = 0.5 * WN * QN;
	vec3 A2DPDT = m_force(tphalfdt, XN, QN, PN, LN, RN, VN, WN);
	vec3 A2DLDT = m_torque(tphalfdt, XN, QN, PN, LN, RN, VN, WN);
	XN = m_X + halfdt * A2DXDT;
	QN.x = m_Q.x + halfdt * A2DQDT.x;
	QN.y = m_Q.y + halfdt * A2DQDT.y;
	QN.z = m_Q.z + halfdt * A2DQDT.z;
	QN.w = m_Q.w + halfdt * A2DQDT.w;
	PN = m_P + halfdt * A2DPDT;
	LN = m_L + halfdt * A2DLDT;
	Convert(QN, PN, LN, RN, VN, WN);

	// A3 = G(t + dt / 2,B2), B3 = S0 + dt * A3
	vec3 A3DXDT = VN;
	quat A3DQDT = 0.5 * WN * QN;
	vec3 A3DPDT = m_force(tphalfdt, XN, QN, PN, LN, RN, VN, WN);
	vec3 A3DLDT = m_torque(tphalfdt, XN, QN, PN, LN, RN, VN, WN);
	XN = m_X + dt * A3DXDT;
	QN.x = m_Q.x + halfdt * A3DQDT.x;
	QN.y = m_Q.y + halfdt * A3DQDT.y;
	QN.z = m_Q.z + halfdt * A3DQDT.z;
	QN.w = m_Q.w + halfdt * A3DQDT.w;
	PN = m_P + dt * A3DPDT;
	LN = m_L + dt * A3DLDT;
	Convert(QN, PN, LN, RN, VN, WN);

	// A4 = G(t + dt, B3),
	// S1 = S0 + (dt / 6) * (A1 + 2 * A2 + 2 * A3 + A4)
	vec3 A4DXDT = VN;
	quat A4DQDT = 0.5 * WN * QN;
	vec3 A4DPDT = m_force(tpdt, XN, QN, PN, LN, RN, VN, WN);
	vec3 A4DLDT = m_torque(tpdt, XN, QN, PN, LN, RN, VN, WN);
	m_X = m_X + sixthdt *(A1DXDT + 2.0*(A2DXDT + A3DXDT) + A4DXDT);
	m_Q.x = m_Q.x + sixthdt *(A1DQDT.x + 2.0*(A2DQDT.x + A3DQDT.x) + A4DQDT.x);
	m_Q.y = m_Q.y + sixthdt *(A1DQDT.y + 2.0*(A2DQDT.y + A3DQDT.y) + A4DQDT.y);
	m_Q.z = m_Q.z + sixthdt *(A1DQDT.z + 2.0*(A2DQDT.z + A3DQDT.z) + A4DQDT.z);
	m_Q.w = m_Q.w + sixthdt *(A1DQDT.w + 2.0*(A2DQDT.w + A3DQDT.w) + A4DQDT.w);
	m_P = m_P + sixthdt *(A1DPDT + 2.0*(A2DPDT + A3DPDT) + A4DPDT);
	m_L = m_L + sixthdt *(A1DLDT + 2.0*(A2DLDT + A3DLDT) + A4DLDT);
	Convert(m_Q, m_P, m_L, m_R, m_V, m_W);
}

std::vector<Particle> spaceships;
std::vector<glm::vec3> shipPath;

void keyboard(unsigned char key, int x, int y)
{
	float angleSpeed = 0.1f;
	float moveSpeed = 0.1f;
	switch(key)
	{
	case 'z': cameraAngle -= angleSpeed; break;
	case 'x': cameraAngle += angleSpeed; break;
	case 'w': cameraPos += cameraDir * moveSpeed; break;
	case 's': cameraPos -= cameraDir * moveSpeed; break;
	case 'd': cameraPos += glm::cross(cameraDir, glm::vec3(0,1,0)) * moveSpeed; break;
	case 'a': cameraPos -= glm::cross(cameraDir, glm::vec3(0,1,0)) * moveSpeed; break;
	}
}

glm::mat4 createCameraMatrix()
{
	// Obliczanie kierunku patrzenia kamery (w plaszczyznie x-z) przy uzyciu zmiennej cameraAngle kontrolowanej przez klawisze.
	cameraDir = glm::vec3(cosf(cameraAngle), 0.0f, sinf(cameraAngle));
	glm::vec3 up = glm::vec3(0,1,0);

	return Core::createViewMatrix(cameraPos, cameraDir, up);
}

void drawObjectColor(obj::Model * model, glm::mat4 modelMatrix, glm::vec3 color)
{
	GLuint program = programColor;

	glUseProgram(program);

	glUniform3f(glGetUniformLocation(program, "objectColor"), color.x, color.y, color.z);
	glUniform3f(glGetUniformLocation(program, "lightDir"), lightDir.x, lightDir.y, lightDir.z);

	glm::mat4 transformation = perspectiveMatrix * cameraMatrix * modelMatrix;
	glUniformMatrix4fv(glGetUniformLocation(program, "modelViewProjectionMatrix"), 1, GL_FALSE, (float*)&transformation);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, GL_FALSE, (float*)&modelMatrix);

	Core::DrawModel(model);

	glUseProgram(0);
}

void drawObjectTexture(obj::Model * model, glm::mat4 modelMatrix, glm::vec3 color)
{
	GLuint program = programTexture;

	glUseProgram(program);

	glUniform3f(glGetUniformLocation(program, "objectColor"), color.x, color.y, color.z);
	glUniform3f(glGetUniformLocation(program, "lightDir"), lightDir.x, lightDir.y, lightDir.z);

	glm::mat4 transformation = perspectiveMatrix * cameraMatrix * modelMatrix;
	glUniformMatrix4fv(glGetUniformLocation(program, "modelViewProjectionMatrix"), 1, GL_FALSE, (float*)&transformation);
	glUniformMatrix4fv(glGetUniformLocation(program, "modelMatrix"), 1, GL_FALSE, (float*)&modelMatrix);

	Core::DrawModel(model);

	glUseProgram(0);
}


 

std::vector<glm::vec3> generatePoints(glm::vec4 startingPoint, glm::vec4 endingPoint, glm::vec4 startingVector, glm::vec4 endingVector, float diff) {
	std::vector<glm::vec3> points;
	for (float i = 0; i < 1; i += diff) {
		glm::vec3 hermiteVec = glm::hermite(startingPoint, endingPoint, startingVector, endingVector, i);
		points.push_back(hermiteVec);
		printf("hermite vect %f %f %f\n", hermiteVec.x, hermiteVec.y, hermiteVec.z);
	}
	printf("size %d generated", points.size());
	return points;
}

/*
void moveObject(obj::Model * model, std::vector<glm::vec3> points)
{
	//float *tab =  new float[sumOfPoints * 4];
	if(t < points.size()) {
		drawObjectTexture(model, , glm::vec3(0.1f, 0.0f, 0.0f));
	} else {
		t = 0;
		drawObjectTexture(model, glm::translate(points[t]), glm::vec3(0.1f, 0.0f, 0.0f));
	}
	/*int z = 0;
	for (int p = 0; p < sumOfPoints; p++) {
		tab[z] = point[i].x;
		z++;
		tab[z] = point[i].y;
		z++;
		tab[z] = point[i].z;
		z++;
		tab[z] = point[i].w;
		z++;
	}
	Core::DrawVertexArray(tab, 4, sumOfPoints * 4);
 
}
*/


glm::vec3 separationV2(Particle x)
{
	glm::vec3 v2 = glm::vec3(0.0f);
	for (int i = 0; i < spaceships.size(); i++)
	{
		if (glm::length(x.pos - spaceships[i].pos)<2)
		{
			v2 = v2 - (x.pos - spaceships[i].pos);
		}
	}
	return v2;
}

void renderScene()
{

	// Aktualizacja macierzy widoku i rzutowania. Macierze sa przechowywane w zmiennych globalnych, bo uzywa ich funkcja drawObject.
	// (Bardziej elegancko byloby przekazac je jako argumenty do funkcji, ale robimy tak dla uproszczenia kodu.
	//  Jest to mozliwe dzieki temu, ze macierze widoku i rzutowania sa takie same dla wszystkich obiektow!)
	cameraMatrix = createCameraMatrix();
	perspectiveMatrix = Core::createPerspectiveMatrix();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0f, 0.3f, 0.3f, 1.0f);
	//printf(" %f %i %f %f %f \n", t, shipPath.size(), shipPath[t % shipPath.size()].x, shipPath[t % shipPath.size()].y, shipPath[t % shipPath.size()].z);
	// Macierz statku "przyczepia" go do kamery. Warto przeanalizowac te linijke i zrozumiec jak to dziala.

	glm::mat4 shipModelMatrix = glm::translate(shipPath[t % shipPath.size()]) * glm::rotate(-cameraAngle + glm::radians(90.0f), glm::vec3(0,1,0)) * glm::scale(glm::vec3(0.25f));
	//printf("shipPath current y: %d\n", shipPath[t % shipPath.size()].y);
	drawObjectColor(&shipModel, shipModelMatrix, glm::vec3(0.6f));

	//drawObjectTexture(&sphereModel, glm::translate(glm::vec3(2,0,2)), glm::vec3(0.8f, 0.2f, 0.3f));
	drawObjectTexture(&sphereModel, glm::translate(glm::vec3(-2,0,-2)), glm::vec3(0.1f, 0.4f, 0.7f));
 
	//moveObject(&shipModel,shipPath);
	glm::vec3 sum = glm::vec3(0);
	for (int i = 0; i < spaceships.size(); i++)
	{
		sum += spaceships[i].vel;
	}


	for (int i = 0; i < spaceships.size(); i++)
	{
		float weight1 = 0.1;
		float weight2 = 0.0001;
		float weight3 = 0.001;


		glm::vec3 v1Attract = glm::normalize( shipPath[t % shipPath.size()] - spaceships[i].pos);
		glm::vec3 v2Seprataion = separationV2(spaceships[i]);
		glm::vec3 v3Aligment = (sum / spaceships.size()) - spaceships[i].vel;
		spaceships[i].vel += (weight1 * v1Attract) + (weight2 * v2Seprataion) + (weight3 * v3Aligment);

		spaceships[i].pos += spaceships[i].vel;

		glm::mat4 shipEnemy = glm::translate(spaceships[i].pos) * glm::scale(glm::vec3(0.25f));
		drawObjectColor(&shipModel, shipEnemy, glm::vec3(0.65f));
	}
	t++;
	glutSwapBuffers();
}

void init()
{
	glEnable(GL_DEPTH_TEST);
	programColor = shaderLoader.CreateProgram("shaders/shader_color.vert", "shaders/shader_color.frag");
	programTexture = shaderLoader.CreateProgram("shaders/shader_tex.vert", "shaders/shader_tex.frag");
	sphereModel = obj::loadModelFromFile("models/sphere.obj");
	shipModel = obj::loadModelFromFile("models/spaceship.obj");
	shipPath = generatePoints(glm::vec4(0, 0, 0, 0), glm::vec4(0, 0, 1, 0), glm::vec4(0, 1 , 0, 0), glm::vec4(10, -7, 0, 0), 0.005);
	printf("shipPath size: %d", shipPath.size());

	for (int i = 0;i<150;i++)
	{
		Particle enemy;
		//gestosc 1+i / 30
		enemy.pos = glm::vec3((1+i/250), 1, 1);
		spaceships.push_back(enemy);
	}

	const int m = 150;
	RigidBody * body[m];
	double time = 1;
	double dt = 1;

	for (int i = 0; i < m; i++)
	{
		body[i] = new RigidBody();
	}
	for (int i = 0; i < m; i++)
	{
		body[i]->Update(t, dt);
		t += dt;
	}

}

void shutdown()
{
	shaderLoader.DeleteProgram(programColor);
	shaderLoader.DeleteProgram(programTexture);
}

void idle()
{
	glutPostRedisplay();
}

int main(int argc, char ** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(200, 200);
	glutInitWindowSize(600, 600);
	glutCreateWindow("OpenGL Pierwszy Program");
	glewInit();

	init();
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(renderScene);
	glutIdleFunc(idle);

	glutMainLoop();

	shutdown();

	return 0;
}
