let temperature = 20;
const minTemp = -250;
const maxTemp = 3000;

const substances = [
  { name: "Oxygen", melt: -219, boil: -183, color: [90, 190, 255], size: 16, mass: 0.75 },
  { name: "Ethanol", melt: -114, boil: 78, color: [255, 190, 80], size: 20, mass: 0.9 },
  { name: "Nuoc", melt: 0, boil: 100, color: [60, 140, 255], size: 22, mass: 1.0 },
  { name: "Thuy ngan", melt: -39, boil: 357, color: [185, 190, 200], size: 24, mass: 1.8 },
  { name: "Sat", melt: 1536, boil: 2880, color: [165, 85, 55], size: 28, mass: 2.2 }
];

let samples = [];

function setup() {
  createCanvas(1200, 760);
  textFont("Arial");

  for (let i = 0; i < substances.length; i++) {
    samples.push(new Sample(substances[i], 90 + i * 215, 250, 175, 250));
  }
}

function draw() {
  drawEnvironment();
  drawThermometer();
  drawPhaseScale();

  for (let sample of samples) {
    sample.update();
    sample.display();
  }

  drawHud();
  drawControls();
}

function drawEnvironment() {
  let t = map(temperature, minTemp, maxTemp, 0, 1);
  let cold = color(20, 45, 120);
  let mild = color(80, 150, 200);
  let hot = color(245, 95, 30);
  let extreme = color(255, 230, 80);

  let bg;
  if (t < 0.5) bg = lerpColor(cold, mild, t / 0.5);
  else if (t < 0.85) bg = lerpColor(mild, hot, (t - 0.5) / 0.35);
  else bg = lerpColor(hot, extreme, (t - 0.85) / 0.15);

  background(bg);

  noStroke();
  for (let y = 0; y < height; y += 8) {
    let a = map(y, 0, height, 45, 0);
    fill(255, 255, 255, a);
    rect(0, y, width, 8);
  }

  fill(0, 120);
  rect(0, 0, width, height);
}

function drawThermometer() {
  const x = 42;
  const y = 90;
  const h = 560;

  stroke(255);
  strokeWeight(2);
  noFill();
  rect(x, y, 24, h, 12);

  let fillH = map(temperature, minTemp, maxTemp, 0, h);
  noStroke();
  fill(temperature < 0 ? color(70, 180, 255) : temperature < 1000 ? color(255, 120, 60) : color(255, 230, 70));
  rect(x + 4, y + h - fillH, 16, fillH, 8);

  fill(255);
  textSize(15);
  textAlign(LEFT, CENTER);
  text(maxTemp + " C", x + 34, y);
  text(minTemp + " C", x + 34, y + h);
  textSize(26);
  textStyle(BOLD);
  text(nf(temperature, 1, 0) + " °C", x - 12, y - 40);
  textStyle(NORMAL);
}

function drawPhaseScale() {
  const x = 110;
  const y = 665;
  const w = 980;
  const h = 22;

  noStroke();
  fill(255, 230);
  textSize(14);
  textAlign(LEFT, BOTTOM);
  text("Thang nhiet do: moi vach la moc nong chay / soi cua tung chat", x, y - 10);

  stroke(255);
  strokeWeight(1);
  noFill();
  rect(x, y, w, h);

  noStroke();
  for (let i = 0; i < substances.length; i++) {
    let s = substances[i];
    let c = color(s.color[0], s.color[1], s.color[2]);

    let mx = map(s.melt, minTemp, maxTemp, x, x + w);
    let bx = map(s.boil, minTemp, maxTemp, x, x + w);

    stroke(c);
    strokeWeight(3);
    line(mx, y - 5, mx, y + h + 5);
    line(bx, y - 5, bx, y + h + 5);

    noStroke();
    fill(c);
    textSize(11);
    textAlign(CENTER, TOP);
    text("NC " + s.melt, mx, y + h + 7);
    text("S " + s.boil, bx, y + h + 20);
  }

  let tx = map(temperature, minTemp, maxTemp, x, x + w);
  stroke(255);
  strokeWeight(4);
  line(tx, y - 12, tx, y + h + 38);
  noStroke();
  fill(255);
  triangle(tx, y - 16, tx - 7, y - 28, tx + 7, y - 28);
}

function drawHud() {
  const x = 705;
  const y = 42;
  const rowH = 42;

  fill(0, 150);
  noStroke();
  rect(x - 16, y - 28, 455, 250, 8);

  fill(255);
  textAlign(LEFT, CENTER);
  textSize(18);
  textStyle(BOLD);
  text("Bang du lieu chuyen the", x, y - 7);
  textStyle(NORMAL);
  textSize(13);
  text("Chat", x, y + 24);
  text("Nong chay", x + 145, y + 24);
  text("Soi", x + 245, y + 24);
  text("Trang thai hien tai", x + 325, y + 24);

  for (let i = 0; i < substances.length; i++) {
    let s = substances[i];
    let yy = y + 55 + i * rowH;
    let phase = getPhase(s);

    fill(s.color[0], s.color[1], s.color[2]);
    circle(x + 8, yy, 15);

    fill(255);
    textSize(14);
    textAlign(LEFT, CENTER);
    text(s.name, x + 22, yy);
    text(s.melt + " °C", x + 145, yy);
    text(s.boil + " °C", x + 245, yy);

    fill(phaseColor(phase));
    text(phaseLabel(phase), x + 325, yy);
  }
}

function drawControls() {
  fill(255, 235);
  textSize(15);
  textAlign(CENTER, CENTER);
  text("Mui ten LEN/XUONG: tang/giam 10 °C   |   TRAI/PHAI: giam/tang 100 °C", width / 2, 726);
}

function keyPressed() {
  if (keyCode === UP_ARROW) temperature += 10;
  if (keyCode === DOWN_ARROW) temperature -= 10;
  if (keyCode === RIGHT_ARROW) temperature += 100;
  if (keyCode === LEFT_ARROW) temperature -= 100;
  temperature = constrain(temperature, minTemp, maxTemp);
}

function getPhase(s) {
  if (temperature < s.melt) return "solid";
  if (temperature < s.boil) return "liquid";
  return "gas";
}

function phaseLabel(p) {
  if (p === "solid") return "Ran";
  if (p === "liquid") return "Long";
  return "Khi";
}

function phaseColor(p) {
  if (p === "solid") return color(210, 230, 255);
  if (p === "liquid") return color(90, 220, 255);
  return color(255, 235, 110);
}

class Sample {
  constructor(data, x, y, w, h) {
    this.data = data;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.particles = [];

    for (let i = 0; i < 32; i++) {
      this.particles.push({
        x: random(x + 35, x + w - 35),
        y: random(y + 70, y + h - 35),
        vx: random(-1, 1),
        vy: random(-1, 1),
        ox: x + 35 + (i % 8) * 15,
        oy: y + 122 + floor(i / 8) * 18
      });
    }
  }

  update() {
    let phase = getPhase(this.data);
    let heat = map(temperature, minTemp, maxTemp, 0.2, 4.4);

    for (let i = 0; i < this.particles.length; i++) {
      let p = this.particles[i];

      if (phase === "solid") {
        let vibration = map(temperature, minTemp, this.data.melt, 0.3, 4.0, true);
        p.x = lerp(p.x, p.ox + sin(frameCount * 0.08 + i) * vibration, 0.22);
        p.y = lerp(p.y, p.oy + cos(frameCount * 0.09 + i) * vibration, 0.22);
      }

      if (phase === "liquid") {
        let speed = map(temperature, this.data.melt, this.data.boil, 0.45, 2.2, true);
        p.vx += random(-0.12, 0.12) * speed;
        p.vy += random(-0.08, 0.08) * speed + 0.025 * this.data.mass;
        p.vx *= 0.96;
        p.vy *= 0.96;
        p.x += p.vx;
        p.y += p.vy;

        let floorY = this.y + this.h - 28;
        let left = this.x + 22;
        let right = this.x + this.w - 22;

        if (p.x < left || p.x > right) p.vx *= -0.9;
        if (p.y > floorY) {
          p.y = floorY;
          p.vy *= -0.45;
          p.vx += random(-0.7, 0.7);
        }
        if (p.y < this.y + 82) p.vy += 0.15;

        p.x = constrain(p.x, left, right);
      }

      if (phase === "gas") {
        let speed = map(temperature, this.data.boil, maxTemp, 1.2, 5.0, true) / this.data.mass;
        p.vx += random(-0.22, 0.22) * speed;
        p.vy += random(-0.22, 0.22) * speed - 0.035 * speed;
        p.vx = constrain(p.vx, -speed * 2.2, speed * 2.2);
        p.vy = constrain(p.vy, -speed * 2.2, speed * 2.2);
        p.x += p.vx;
        p.y += p.vy;

        let left = this.x + 8;
        let right = this.x + this.w - 8;
        let top = this.y + 52;
        let bottom = this.y + this.h - 12;

        if (p.x < left || p.x > right) p.vx *= -1;
        if (p.y < top || p.y > bottom) p.vy *= -1;
        p.x = constrain(p.x, left, right);
        p.y = constrain(p.y, top, bottom);
      }

      p.vx += random(-0.01, 0.01) * heat;
      p.vy += random(-0.01, 0.01) * heat;
    }
  }

  display() {
    let s = this.data;
    let phase = getPhase(s);

    noStroke();
    fill(0, 120);
    rect(this.x, this.y, this.w, this.h, 8);

    stroke(255, 90);
    noFill();
    rect(this.x, this.y, this.w, this.h, 8);

    fill(255);
    noStroke();
    textAlign(CENTER, CENTER);
    textSize(18);
    textStyle(BOLD);
    text(s.name, this.x + this.w / 2, this.y + 22);
    textStyle(NORMAL);

    fill(255, 220);
    textSize(12);
    text("NC: " + s.melt + " °C  |  Soi: " + s.boil + " °C", this.x + this.w / 2, this.y + 43);

    fill(phaseColor(phase));
    textSize(16);
    text(phaseLabel(phase), this.x + this.w / 2, this.y + 66);

    if (phase === "liquid") {
      let liquidLevel = this.y + this.h - 70;
      fill(s.color[0], s.color[1], s.color[2], 65);
      noStroke();
      rect(this.x + 15, liquidLevel, this.w - 30, this.y + this.h - liquidLevel - 15, 0, 0, 8, 8);
      stroke(255, 150);
      noFill();
      beginShape();
      for (let xx = this.x + 15; xx <= this.x + this.w - 15; xx += 8) {
        let yy = liquidLevel + sin(frameCount * 0.06 + xx * 0.06) * 4;
        vertex(xx, yy);
      }
      endShape();
    }

    if (phase === "gas") {
      noFill();
      stroke(s.color[0], s.color[1], s.color[2], 90);
      for (let i = 0; i < 5; i++) {
        let yy = this.y + 95 + i * 24;
        arc(this.x + this.w / 2, yy, 55 + i * 10, 18, PI, TWO_PI);
      }
    }

    for (let p of this.particles) {
      noStroke();
      let alpha = phase === "gas" ? 150 : 230;
      fill(s.color[0], s.color[1], s.color[2], alpha);

      let r = s.size;
      if (phase === "solid") r *= 0.72;
      if (phase === "liquid") r *= 0.62;
      if (phase === "gas") r *= 0.42;

      circle(p.x, p.y, r);

      fill(255, phase === "gas" ? 70 : 110);
      circle(p.x - r * 0.15, p.y - r * 0.15, r * 0.32);
    }

    this.drawMiniPhaseBar();
  }

  drawMiniPhaseBar() {
    let s = this.data;
    let bx = this.x + 18;
    let by = this.y + this.h - 18;
    let bw = this.w - 36;
    let bh = 8;

    noStroke();
    fill(210, 230, 255);
    rect(bx, by, bw * 0.33, bh);
    fill(90, 220, 255);
    rect(bx + bw * 0.33, by, bw * 0.34, bh);
    fill(255, 235, 110);
    rect(bx + bw * 0.67, by, bw * 0.33, bh);

    let localMin = minTemp;
    let localMax = maxTemp;
    let m = map(s.melt, localMin, localMax, bx, bx + bw);
    let b = map(s.boil, localMin, localMax, bx, bx + bw);
    let t = map(temperature, localMin, localMax, bx, bx + bw);

    stroke(255);
    strokeWeight(2);
    line(m, by - 4, m, by + bh + 4);
    line(b, by - 4, b, by + bh + 4);

    stroke(255, 40, 40);
    strokeWeight(3);
    line(t, by - 7, t, by + bh + 7);
  }
}
