/* 
 * File:   maze.cpp
 * Author: yky
 *
 * Created on April 11, 2016, 10:15 PM
 */

#include <cstdlib>

using namespace std;

// A game where you guide a moving square through a maze and try to avoid collisions.

#include <iostream>
#include <random>
#include <sstream>
#include <SFML/Graphics.hpp>

const char* levels[] = {

        "7 5"

        "..+E+.."
        "..+.+.."
        "..+.+.."
        "..+.+.."
        "..+S+.."

        ,

        "7 5"

        "...+..."
        ".+.+.+."
        ".+.+.+."
        ".+.+.+."
        "S+...+E"

        ,

        "10 6"

        "++...++..."
        "...+.++.+."
        ".+++.++.+."
        ".+...+..+."
        ".+.+++.+.."
        "S+.....+E+"

        ,

        "14 9"

        ".+++....+....."
        ".....++...+.+."
        ".+++...+..+.+."
        ".....+++++...."
        ".+++++S+.+..+."
        ".+.....+...++."
        ".+.+++++..++.."
        ".+.+E..+.....+"
        "...+++...+++++"

        ,

        "20 11"

        "++..........+...++++"
        "+++..++++++.+.+.+..."
        "...+..+E..+.+.+...+."
        ".+..+..++.+...+..+.."
        "..+..+..+..+++++...+"
        "+..+..+..+....+..+++"
        ".+..+..+..+..+..+..."
        "..+..+..+...+..++.+."
        "..S+..+..+++++....+."
        ".++++..+...+++++++.."
        ".......+++.........+"

        ,

        "23 16"

        ".......+..............."
        "...+...+.+++++++++++++."
        "...++..+.............+."
        "..+..+.+++++++++++++.+."
        ".+...+.+E..........+.+."
        ".+.+.+.++++++++....+.+."
        ".+.+.+.+.......+...+.+."
        ".+.+.+..+.S+...+...+.+."
        ".+.+.+...+++...+...+.+."
        ".+.+..++......+....+.+."
        ".+.+....++++++.....+.+."
        ".+.+...............+.+."
        ".+.+++++++++++++++++.+."
        ".+...................+."
        ".+++++++++++++++++++++."
        "......................."

        ,

        "35 21"

        "+.................................."
        ".+..++++++++++++++++++++++++++++..."
        "..+..+.....+...+...+...+.......+..."
        "...+..+..+...+...+...+.+..+++..+..."
        "....+..+..++++++++++++.+..+.E+.+..."
        ".++..+..+..+.+.+.+.+.+.+..+.++.+..."
        ".+++..+..+..+.+.+.+.+.....+....+..."
        ".++++..+..+..+++++++++++++++++++..."
        ".++++...+..+..++..................."
        "....+....+..+..+.++++++++++++++++++"
        "+++.+.....+..+...+................."
        "....+......+..+..+.++++++++++++.++."
        ".++++.......+..+.+.+.....+....+.+.."
        ".+...........+..++.+..+.+++.+.+.+.+"
        ".+..++..++...++..+.+..+..+..+.+.+.."
        ".+...+..+....+.+...+..++.+.++.+.++."
        ".+....++.....+..+..+..+..+..+.+.+.."
        ".+..+....+...+...+.+..+.+++.+.+.+.+"
        ".+..++++++...+.+..++..+..+..+.+.+.."
        ".+...........+.++..+..++.+.++.++++."
        "S+.............+++....+.....+......"

        ,

        "40 10"

        "S........+...........+.......++.....+..."
        "+++++...++..........+..+++++....+++..+.."
        "....+..+..+...++++++...+...+++++..+.+..."
        "...+..+...+..+...+.+...+.+.+......+..+.+"
        "..+..+..+.+.+..+.+.+...+.+.+......+.+.+."
        ".+..+...+.++..+..+.+.E.+.+.+......+..+.."
        "+..++++.+....+..++.+++++.+.+.....+..+..."
        "..+.....++++++...+.....+.+.++++++..+...."
        "....+++++.....++.+++++++.+.+...+..+....."
        "...+...........+.........+...+...+......"

        ,

        "34 12"

        ".................................."
        ".....+...+..+++..+...+............"
        "......+.+..+...+.+...+............"
        ".......+...+...+.+...+............"
        ".......+....+++...+++............."
        ".................................."
        "..............................+..."
        ".....+....+....+..+++..+..+...+..."
        "......+...+...+..+...+.++.+...+..."
        ".......+.+.+.+...+...+.+.++......."
        "........+...+.....+++..+..+...+..."
        ".................................."

        ,

        0

};

class Game {

        public:

        Game();

        void loadLevel();

        void run();
        void update();
        void draw();

        private:

        sf::VideoMode m_videoMode;
        sf::RenderWindow m_window;
        sf::RectangleShape m_pointer;
        std::vector<sf::RectangleShape> m_walls;
        sf::RectangleShape m_start, m_end;

        int m_level;

};

Game::Game():
        m_videoMode(800, 600),
        m_window(m_videoMode, "Tiny Maze", sf::Style::Default & ~sf::Style::Resize),
        m_level(0) {

        m_window.setFramerateLimit(60);

        m_pointer.setFillColor(sf::Color::Black);
        m_pointer.setOutlineColor(sf::Color::White);
        m_pointer.setOutlineThickness(2);

        loadLevel();

}

void Game::loadLevel() {

        m_walls.clear();
        m_start = m_end = sf::RectangleShape();

        std::istringstream level(levels[m_level]);

        int width = 0, height = 0;
        level >> width >> height;

        const int dx = m_videoMode.width / width, dy = m_videoMode.height / height;
        m_window.setSize(sf::Vector2u(dx * width, dy * height));
        m_pointer.setSize(sf::Vector2f(1, 1) * (std::min(dx, dy) * .1f));

        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {

                        char c;
                        level >> c;

                        sf::RectangleShape wall(sf::Vector2f(dx, dy));
                        wall.setPosition(x * dx, y * dy);

                        if (c == '+')
                                m_walls.push_back(wall);
                        else if (c == 'S') {
                                wall.setFillColor(sf::Color::Green);
                                m_start = wall;
                        } else if (c == 'E') {
                                wall.setFillColor(sf::Color::Red);
                                m_end = wall;
                        }

                }
        }

        sf::Mouse::setPosition(sf::Vector2i(m_start.getPosition().x + dx / 2.f, m_start.getPosition().y + dy / 2.f), m_window);

}

void Game::run() {

        while (m_window.isOpen()) {

                update();

                sf::Event event;
                while (m_window.pollEvent(event))
                        if (event.type == sf::Event::Closed
                                or (event.type == sf::Event::KeyPressed and event.key.code == sf::Keyboard::Escape))
                                m_window.close();

                draw();
                m_window.display();

        }

}

void Game::update() {

        const sf::Vector2i mouse(sf::Mouse::getPosition(m_window));
        m_pointer.setPosition(mouse.x - (m_pointer.getSize().x / 2), mouse.y - (m_pointer.getSize().y / 2));

        if (not sf::FloatRect(0, 0, m_window.getSize().x, m_window.getSize().y).intersects(sf::FloatRect(m_pointer.getPosition(), m_pointer.getSize())))
                loadLevel();

        {
                int count = 0;
                for (int i = 0; i < m_pointer.getPointCount(); ++i)
                        count += sf::FloatRect(m_end.getPosition(), m_end.getSize()).contains
                                (m_pointer.getPosition() + m_pointer.getPoint(i));
                if (count == m_pointer.getPointCount())
                        ++m_level, loadLevel();
        }

        for (const auto& wall: m_walls)
                for (int i = 0; i < m_pointer.getPointCount(); ++i)
                        if (sf::FloatRect(wall.getPosition(), wall.getSize()).contains
                                (m_pointer.getPosition() + m_pointer.getPoint(i)))
                                loadLevel();

}

void Game::draw() {

        m_window.clear(sf::Color(50, 50, 50));

        for (const auto& wall: m_walls)
                m_window.draw(wall);

        m_window.draw(m_start);
        m_window.draw(m_end);

        m_window.draw(m_pointer);

}

extern "C" int main2() {
        Game().run();
        return EXIT_SUCCESS;
}
